from __future__ import annotations

import random
from collections import deque

import numpy as np


TEMPLATE_LIBRARY = [
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
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
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    ),
]

CELL_H = 7
CELL_W = 7
REGION_COUNT_RANGE = (2, 4)
GRID_H_RANGE = (26, 40)
GRID_W_RANGE = (26, 40)
INNER_H_RANGE = (5, 10)
INNER_W_RANGE = (5, 11)
PLACEMENT_GAP = 2
GEN_RETRIES = 400


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _outer_border_is_color(
    grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]
) -> bool:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    return bool(
        np.all(grid[r0, c0 : c1 + 1] == color)
        and np.all(grid[r1, c0 : c1 + 1] == color)
        and np.all(grid[r0 : r1 + 1, c0] == color)
        and np.all(grid[r0 : r1 + 1, c1] == color)
    )


def _full_separator_rows(
    grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]
) -> list[int]:
    r0, c0, r1, c1 = bbox
    return [
        r
        for r in range(r0 + 1, r1)
        if np.all(grid[r, c0 + 1 : c1] == color)
    ]


def _full_separator_cols(
    grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]
) -> list[int]:
    r0, c0, r1, c1 = bbox
    return [
        c
        for c in range(c0 + 1, c1)
        if np.all(grid[r0 + 1 : r1, c] == color)
    ]


def _mask_center(mask: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(mask)
    return (
        int((pts[:, 0].min() + pts[:, 0].max()) // 2),
        int((pts[:, 1].min() + pts[:, 1].max()) // 2),
    )


def _extract_shape(cell: np.ndarray, color: int) -> np.ndarray:
    pts = np.argwhere(cell == color)
    assert len(pts) >= 2
    key_col = int(pts[:, 1].min())
    assert int(np.sum(pts[:, 1] == key_col)) == 1
    shape_pts = pts[pts[:, 1] > key_col]
    assert len(shape_pts) > 0
    r0 = int(shape_pts[:, 0].min())
    c0 = int(shape_pts[:, 1].min())
    r1 = int(shape_pts[:, 0].max())
    c1 = int(shape_pts[:, 1].max())
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[shape_pts[:, 0] - r0, shape_pts[:, 1] - c0] = True
    return out


def _parse_legend(
    grid: np.ndarray, bbox: tuple[int, int, int, int], border: int, bg: int
) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = bbox
    sep_rows = _full_separator_rows(grid, border, bbox)
    sep_cols = _full_separator_cols(grid, border, bbox)
    assert bool(sep_rows) ^ bool(sep_cols)

    mapping: dict[int, np.ndarray] = {}
    if sep_cols:
        cuts = [c0] + sep_cols + [c1]
        for left, right in zip(cuts, cuts[1:]):
            cell = grid[r0 + 1 : r1, left + 1 : right]
            colors = set(map(int, np.unique(cell))) - {bg}
            assert len(colors) == 1
            color = next(iter(colors))
            mapping[color] = _extract_shape(cell, color)
    else:
        cuts = [r0] + sep_rows + [r1]
        for top, bottom in zip(cuts, cuts[1:]):
            cell = grid[top + 1 : bottom, c0 + 1 : c1]
            colors = set(map(int, np.unique(cell))) - {bg}
            assert len(colors) == 1
            color = next(iter(colors))
            mapping[color] = _extract_shape(cell, color)

    assert mapping
    return mapping


def _find_border_color(grid: np.ndarray, bg: int) -> int:
    scores: list[tuple[int, int, int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        rect_count = 0
        cell_count = 0
        for comp in _components(grid == color):
            bbox = _bbox(comp)
            if _outer_border_is_color(grid, color, bbox):
                rect_count += 1
                cell_count += len(comp)
        if rect_count:
            scores.append((rect_count, cell_count, color))
    assert scores
    scores.sort()
    return int(scores[-1][2])


def _stamp(
    canvas: np.ndarray, shape: np.ndarray, anchor: tuple[int, int], color: int
) -> None:
    ar, ac = anchor
    cr, cc = _mask_center(shape)
    for rr, cc2 in np.argwhere(shape):
        tr = ar + int(rr) - cr
        tc = ac + int(cc2) - cc
        if 0 <= tr < canvas.shape[0] and 0 <= tc < canvas.shape[1]:
            canvas[tr, tc] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)
    border = _find_border_color(g, bg)

    legend_boxes: list[tuple[int, int, int, int]] = []
    region_boxes: list[tuple[int, int, int, int]] = []
    for comp in _components(g == border):
        bbox = _bbox(comp)
        if not _outer_border_is_color(g, border, bbox):
            continue
        if _full_separator_rows(g, border, bbox) or _full_separator_cols(g, border, bbox):
            legend_boxes.append(bbox)
        else:
            region_boxes.append(bbox)

    assert len(legend_boxes) == 1
    mapping = _parse_legend(g, legend_boxes[0], border, bg)

    out = g.copy()
    for r0, c0, r1, c1 in region_boxes:
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        colors = set(map(int, np.unique(interior))) - {bg, border}
        assert len(colors) == 1
        color = next(iter(colors))
        assert color in mapping

        pts = np.argwhere(interior == color)
        anchor = (
            int((pts[:, 0].min() + pts[:, 0].max()) // 2),
            int((pts[:, 1].min() + pts[:, 1].max()) // 2),
        )
        interior[:, :] = bg
        _stamp(interior, mapping[color], anchor, color)
    return out


def _draw_box(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_legend_cell(cell: np.ndarray, color: int, shape: np.ndarray) -> None:
    key_r = cell.shape[0] // 2
    cell[key_r, 0] = color
    sh, sw = shape.shape
    sr = (cell.shape[0] - sh) // 2
    sc = 2 + (cell.shape[1] - 2 - sw) // 2
    view = cell[sr : sr + sh, sc : sc + sw]
    view[shape] = color


def _build_legend(
    bg: int,
    border: int,
    colors: list[int],
    shapes: list[np.ndarray],
    orientation: str,
) -> np.ndarray:
    n = len(colors)
    if orientation == "top":
        legend = np.full((CELL_H + 2, 2 + n * CELL_W + (n - 1)), bg, dtype=int)
        _draw_box(legend, (0, 0, legend.shape[0] - 1, legend.shape[1] - 1), border)
        cursor = 1
        for idx, (color, shape) in enumerate(zip(colors, shapes)):
            cell = legend[1 : 1 + CELL_H, cursor : cursor + CELL_W]
            _draw_legend_cell(cell, color, shape)
            cursor += CELL_W
            if idx + 1 < n:
                legend[1 : 1 + CELL_H, cursor] = border
                cursor += 1
        return legend

    legend = np.full((2 + n * CELL_H + (n - 1), CELL_W + 2), bg, dtype=int)
    _draw_box(legend, (0, 0, legend.shape[0] - 1, legend.shape[1] - 1), border)
    cursor = 1
    for idx, (color, shape) in enumerate(zip(colors, shapes)):
        cell = legend[cursor : cursor + CELL_H, 1 : 1 + CELL_W]
        _draw_legend_cell(cell, color, shape)
        cursor += CELL_H
        if idx + 1 < n:
            legend[cursor, 1 : 1 + CELL_W] = border
            cursor += 1
    return legend


def _boxes_conflict(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_box(
    rng: random.Random,
    taken: list[tuple[int, int, int, int]],
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int] | None:
    if row_lo > row_hi or col_lo > col_hi:
        return None
    for _ in range(200):
        r0 = rng.randint(row_lo, row_hi)
        c0 = rng.randint(col_lo, col_hi)
        box = (r0, c0, r0 + height - 1, c0 + width - 1)
        if all(not _boxes_conflict(box, other, PLACEMENT_GAP) for other in taken):
            return box
    return None


def _anchor_candidates(
    inner_h: int, inner_w: int, shape: np.ndarray, want_clip: bool
) -> list[tuple[int, int]]:
    pts = np.argwhere(shape)
    cr, cc = _mask_center(shape)
    min_dr = int(np.min(pts[:, 0] - cr))
    max_dr = int(np.max(pts[:, 0] - cr))
    min_dc = int(np.min(pts[:, 1] - cc))
    max_dc = int(np.max(pts[:, 1] - cc))

    out: list[tuple[int, int]] = []
    for r in range(inner_h):
        for c in range(inner_w):
            clipped = (
                r + min_dr < 0
                or r + max_dr >= inner_h
                or c + min_dc < 0
                or c + max_dc >= inner_w
            )
            if clipped == want_clip:
                out.append((r, c))
    return out


def _make_fragments(
    rng: random.Random, inner_h: int, inner_w: int, anchor: tuple[int, int]
) -> list[tuple[int, int]]:
    ar, ac = anchor
    span_r = rng.randint(0, min(2, ar, inner_h - 1 - ar))
    span_c = rng.randint(0, min(2, ac, inner_w - 1 - ac))
    pts = {(ar, ac)}
    if span_r:
        pts.add((ar - span_r, ac))
        pts.add((ar + span_r, ac))
    if span_c:
        pts.add((ar, ac - span_c))
        pts.add((ar, ac + span_c))

    extras = [
        (r, c)
        for r in range(ar - span_r, ar + span_r + 1)
        for c in range(ac - span_c, ac + span_c + 1)
        if (r, c) not in pts
    ]
    rng.shuffle(extras)
    target = rng.randint(len(pts), min(len(pts) + 3, len(pts) + len(extras)))
    for pt in extras[: target - len(pts)]:
        pts.add(pt)
    return sorted(pts)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(["top", "left"])
    bg = rng.randrange(10)
    border_choices = [c for c in range(10) if c != bg]
    border = rng.choice(border_choices)
    palette = [c for c in range(10) if c not in (bg, border)]
    color_count = rng.randint(2, min(4, len(palette)))
    colors = rng.sample(palette, color_count)
    shapes = [shape.copy() for shape in rng.sample(TEMPLATE_LIBRARY, color_count)]
    legend = _build_legend(bg, border, colors, shapes, orientation)

    if orientation == "top":
        min_h = max(GRID_H_RANGE[0], legend.shape[0] + 12)
        min_w = max(GRID_W_RANGE[0], legend.shape[1] + 2)
        if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
            return None
        height = rng.randint(min_h, GRID_H_RANGE[1])
        width = rng.randint(min_w, GRID_W_RANGE[1])
        legend_r = 0
        legend_c = rng.randint(0, width - legend.shape[1])
        row_lo = legend.shape[0] + 1
        col_lo = 1
    else:
        min_h = max(GRID_H_RANGE[0], legend.shape[0] + 2)
        min_w = max(GRID_W_RANGE[0], legend.shape[1] + 12)
        if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
            return None
        height = rng.randint(min_h, GRID_H_RANGE[1])
        width = rng.randint(min_w, GRID_W_RANGE[1])
        legend_r = rng.randint(0, height - legend.shape[0])
        legend_c = 0
        row_lo = 1
        col_lo = legend.shape[1] + 1

    grid = np.full((height, width), bg, dtype=int)
    grid[
        legend_r : legend_r + legend.shape[0], legend_c : legend_c + legend.shape[1]
    ] = legend
    taken = [
        (
            legend_r,
            legend_c,
            legend_r + legend.shape[0] - 1,
            legend_c + legend.shape[1] - 1,
        )
    ]

    region_count = rng.randint(REGION_COUNT_RANGE[0], min(REGION_COUNT_RANGE[1], color_count))
    region_colors = rng.sample(colors, region_count)
    shape_by_color = {color: shape for color, shape in zip(colors, shapes)}
    clip_flags = [i % 2 == 0 for i in range(region_count)]
    rng.shuffle(clip_flags)

    specs = []
    for color, want_clip in zip(region_colors, clip_flags):
        shape = shape_by_color[color]
        inner_h = rng.randint(max(INNER_H_RANGE[0], shape.shape[0]), INNER_H_RANGE[1])
        inner_w = rng.randint(max(INNER_W_RANGE[0], shape.shape[1]), INNER_W_RANGE[1])
        specs.append(
            {
                "color": color,
                "shape": shape,
                "want_clip": want_clip,
                "inner_h": inner_h,
                "inner_w": inner_w,
            }
        )

    specs.sort(key=lambda item: item["inner_h"] * item["inner_w"], reverse=True)
    placed = []
    for spec in specs:
        outer_h = spec["inner_h"] + 2
        outer_w = spec["inner_w"] + 2
        row_hi = height - outer_h - 1
        col_hi = width - outer_w - 1
        box = _place_box(rng, taken, row_lo, row_hi, col_lo, col_hi, outer_h, outer_w)
        if box is None:
            return None
        taken.append(box)
        placed.append((box, spec))

    for box, spec in placed:
        _draw_box(grid, box, border)
        r0, c0, r1, c1 = box
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        candidates = _anchor_candidates(
            inner.shape[0], inner.shape[1], spec["shape"], spec["want_clip"]
        )
        if not candidates:
            candidates = _anchor_candidates(
                inner.shape[0], inner.shape[1], spec["shape"], not spec["want_clip"]
            )
        if not candidates:
            return None
        anchor = rng.choice(candidates)
        for rr, cc in _make_fragments(rng, inner.shape[0], inner.shape[1], anchor):
            inner[rr, cc] = spec["color"]

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid grid")
