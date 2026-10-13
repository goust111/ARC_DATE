from __future__ import annotations

import random

import numpy as np


def _border_bg(grid: np.ndarray) -> int:
    h, w = grid.shape
    border_parts = [grid[0, :], grid[-1, :]]
    if h > 2:
        border_parts.append(grid[1:-1, 0])
        border_parts.append(grid[1:-1, -1])
    border = np.concatenate(border_parts)
    colors, counts = np.unique(border, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _extract_marker_order(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    row_cols = np.flatnonzero(grid[-1, :] != bg)
    col_rows = np.flatnonzero(grid[:, -1] != bg)

    if row_cols.size > col_rows.size:
        assert row_cols.size >= 2, "bottom legend must contain at least two markers"
        order = [int(grid[-1, c]) for c in row_cols]
        scene = grid.copy()
        scene[-1, :] = bg
    else:
        assert col_rows.size >= 2, "right legend must contain at least two markers"
        order = [int(grid[r, -1]) for r in col_rows]
        scene = grid.copy()
        scene[:, -1] = bg

    assert len(order) == len(set(order)), "marker colors must be unique"
    return order, scene


def _parse_rectangles(scene: np.ndarray, bg: int) -> dict[int, dict[str, tuple[int, int] | int]]:
    rects: dict[int, dict[str, tuple[int, int] | int]] = {}
    for coords in _components(scene != bg):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, c0 = int(rows.min()), int(cols.min())
        r1, c1 = int(rows.max()), int(cols.max())
        sub = scene[r0 : r1 + 1, c0 : c1 + 1]
        assert int(coords.shape[0]) == int(sub.size), "each object must be a filled rectangle"

        colors, counts = np.unique(sub, return_counts=True)
        assert len(colors) == 2, "each rectangle must contain exactly one anchor color"
        anchor_candidates = [int(color) for color, count in zip(colors.tolist(), counts.tolist(), strict=False) if int(count) == 1]
        assert len(anchor_candidates) == 1, "each rectangle must contain one singleton anchor pixel"
        anchor_color = anchor_candidates[0]
        fill_colors = [int(color) for color in colors.tolist() if int(color) != anchor_color]
        assert len(fill_colors) == 1
        fill_color = fill_colors[0]

        anchor_pos = np.argwhere(sub == anchor_color)
        assert anchor_pos.shape == (1, 2)
        ar, ac = map(int, anchor_pos[0])
        assert fill_color not in rects, "rectangle colors must be unique"
        rects[fill_color] = {
            "shape": (int(sub.shape[0]), int(sub.shape[1])),
            "anchor": (ar, ac),
            "fill": fill_color,
        }

    assert rects, "expected at least one rectangle"
    return rects


def _stack_rectangles(
    order: list[int],
    rects: dict[int, dict[str, tuple[int, int] | int]],
    out_shape: tuple[int, int],
    bg: int,
) -> np.ndarray:
    out = np.full(out_shape, bg, dtype=int)
    center_r = out_shape[0] // 2
    center_c = out_shape[1] // 2

    for color in order:
        record = rects[color]
        rect_h, rect_w = record["shape"]  # type: ignore[index]
        anchor_r, anchor_c = record["anchor"]  # type: ignore[index]
        fill = int(record["fill"])  # type: ignore[index]

        top = center_r - int(anchor_r)
        left = center_c - int(anchor_c)
        dst_r0 = max(0, top)
        dst_c0 = max(0, left)
        dst_r1 = min(out_shape[0], top + int(rect_h))
        dst_c1 = min(out_shape[1], left + int(rect_w))
        if dst_r0 < dst_r1 and dst_c0 < dst_c1:
            out[dst_r0:dst_r1, dst_c0:dst_c1] = fill

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Layer colored rectangles into one centered crop using the edge-marker order and each rectangle's singleton anchor pixel.

    Input structure:
    - The background is black.
    - Several solid rectangles of distinct colors appear in the scene, and they do not overlap.
    - Each rectangle contains exactly one singleton anchor pixel of another color.
    - A row of singleton color markers sits on the bottom edge, or a column of markers sits on the right edge; these colors match the rectangles.

    Rule:
    - Read the edge markers in order: left-to-right on the bottom edge, or top-to-bottom on the right edge. That order is the layer order from bottom to top.
    - Remove the marker edge and parse each remaining non-background object as one filled rectangle with one singleton anchor pixel.
    - Build an output canvas whose height and width match the largest rectangle extents, align every rectangle so its anchor lands on the output center cell, and paint the rectangles in marker order so later ones overwrite earlier ones.
    - The singleton anchor pixels are not preserved; each rectangle contributes only its solid base color, clipped by the output boundary.

    Key insight:
    - The edge markers encode only z-order, while the singleton anchor inside each rectangle encodes translation; the final image is the centered overlay of those translated rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _border_bg(g)
    order, scene = _extract_marker_order(g, bg)
    rects = _parse_rectangles(scene, bg)
    assert set(order) == set(rects), "legend colors must match rectangle colors one-to-one"

    out_h = max(int(record["shape"][0]) for record in rects.values())  # type: ignore[index]
    out_w = max(int(record["shape"][1]) for record in rects.values())  # type: ignore[index]
    return _stack_rectangles(order, rects, (out_h, out_w), bg)


_GEN_BG = 0
_GEN_ANCHOR = 5
_GEN_FILL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_GEN_RECT_COUNT_RANGE = (2, 5)
_GEN_OUTPUT_DIM_MAX = {2: 11, 3: 11, 4: 9, 5: 7}
_GEN_CANVAS_MIN = 16
_GEN_CANVAS_MAX = 30
_GEN_PLACE_TRIES = 120
_GEN_ATTEMPTS = 500


def _odd_choices(lo: int, hi: int) -> list[int]:
    return [value for value in range(lo, hi + 1) if value % 2 == 1]


def _choose_rectangles(rng: np.random.Generator, count: int) -> tuple[tuple[int, int], dict[int, dict[str, tuple[int, int] | int]], list[int]]:
    max_dim = _GEN_OUTPUT_DIM_MAX[count]
    out_h = int(rng.choice(_odd_choices(5, max_dim)))
    out_w = int(rng.choice(_odd_choices(5, max_dim)))

    colors = [int(color) for color in rng.choice(_GEN_FILL_COLORS, size=count, replace=False)]
    order = [int(color) for color in rng.permutation(colors)]

    rects: dict[int, dict[str, tuple[int, int] | int]] = {}
    largest_color = colors[0]
    rects[largest_color] = {
        "shape": (out_h, out_w),
        "anchor": (int(rng.integers(0, out_h)), int(rng.integers(0, out_w))),
        "fill": largest_color,
    }

    for color in colors[1:]:
        while True:
            rect_h = int(rng.integers(2, out_h + 1))
            rect_w = int(rng.integers(2, out_w + 1))
            if rect_h == out_h and rect_w == out_w:
                continue
            rects[color] = {
                "shape": (rect_h, rect_w),
                "anchor": (int(rng.integers(0, rect_h)), int(rng.integers(0, rect_w))),
                "fill": color,
            }
            break

    return (out_h, out_w), rects, order


def _choose_canvas(
    rng: np.random.Generator,
    rects: dict[int, dict[str, tuple[int, int] | int]],
) -> tuple[int, int]:
    padded_area = 0
    for record in rects.values():
        rect_h, rect_w = record["shape"]  # type: ignore[index]
        padded_area += (int(rect_h) + 2) * (int(rect_w) + 2)
    base = max(_GEN_CANVAS_MIN, int(np.ceil(np.sqrt(padded_area))) + 4)
    height = int(rng.integers(base, _GEN_CANVAS_MAX + 1))
    width = int(rng.integers(base, _GEN_CANVAS_MAX + 1))
    return height, width


def _place_rectangles(
    rng: np.random.Generator,
    rects: dict[int, dict[str, tuple[int, int] | int]],
    height: int,
    width: int,
    bottom_markers: bool,
) -> dict[int, tuple[int, int]] | None:
    occupancy = np.zeros((height, width), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}
    reserve_bottom = 1 if bottom_markers else 0
    reserve_right = 0 if bottom_markers else 1

    items = sorted(
        rects.items(),
        key=lambda item: int(item[1]["shape"][0]) * int(item[1]["shape"][1]),  # type: ignore[index]
        reverse=True,
    )
    for color, record in items:
        rect_h, rect_w = record["shape"]  # type: ignore[index]
        rect_h = int(rect_h)
        rect_w = int(rect_w)
        max_r = height - rect_h - 1 - reserve_bottom
        max_c = width - rect_w - 1 - reserve_right
        if max_r < 1 or max_c < 1:
            return None

        for _ in range(_GEN_PLACE_TRIES):
            top = int(rng.integers(1, max_r + 1))
            left = int(rng.integers(1, max_c + 1))
            if occupancy[top - 1 : top + rect_h + 1, left - 1 : left + rect_w + 1].any():
                continue
            occupancy[top : top + rect_h, left : left + rect_w] = True
            placements[color] = (top, left)
            break
        else:
            return None

    return placements


def _draw_input(
    height: int,
    width: int,
    rects: dict[int, dict[str, tuple[int, int] | int]],
    placements: dict[int, tuple[int, int]],
    order: list[int],
    bottom_markers: bool,
    rng: np.random.Generator,
) -> np.ndarray | None:
    grid = np.full((height, width), _GEN_BG, dtype=int)

    for color, record in rects.items():
        rect_h, rect_w = record["shape"]  # type: ignore[index]
        anchor_r, anchor_c = record["anchor"]  # type: ignore[index]
        top, left = placements[color]
        rect_h = int(rect_h)
        rect_w = int(rect_w)
        anchor_r = int(anchor_r)
        anchor_c = int(anchor_c)

        grid[top : top + rect_h, left : left + rect_w] = int(color)
        grid[top + anchor_r, left + anchor_c] = _GEN_ANCHOR

    if bottom_markers:
        if width - 2 < len(order):
            return None
        cols = sorted(int(value) for value in rng.choice(np.arange(1, width - 1), size=len(order), replace=False))
        for color, col in zip(order, cols, strict=False):
            grid[-1, col] = int(color)
    else:
        if height - 2 < len(order):
            return None
        rows = sorted(int(value) for value in rng.choice(np.arange(1, height - 1), size=len(order), replace=False))
        for color, row in zip(order, rows, strict=False):
            grid[row, -1] = int(color)

    return grid


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    rect_count = int(rng.integers(_GEN_RECT_COUNT_RANGE[0], _GEN_RECT_COUNT_RANGE[1] + 1))
    out_shape, rects, order = _choose_rectangles(rng, rect_count)
    height, width = _choose_canvas(rng, rects)
    bottom_markers = bool(rng.integers(0, 2))

    placements = _place_rectangles(rng, rects, height, width, bottom_markers)
    if placements is None:
        return None

    grid = _draw_input(height, width, rects, placements, order, bottom_markers, rng)
    if grid is None:
        return None

    expected = _stack_rectangles(order, rects, out_shape, _GEN_BG)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    visible = set(map(int, np.unique(actual))) - {_GEN_BG}
    if len(visible) < min(3, rect_count):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layering example")
