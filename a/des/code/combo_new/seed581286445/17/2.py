from __future__ import annotations

from collections import Counter
import random

import numpy as np


_BG = 0
_FRAME = 5
_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
_LAYOUTS = [(2, 2), (2, 3), (3, 2)]
_GEN_RETRIES = 400

_BASE_SHAPES = [
    {(0, 0), (1, 0)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (1, 1), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (2, 1)},
]


def _dominant_color(values: np.ndarray) -> int:
    flat = values.reshape(-1)
    counts = Counter(int(v) for v in flat)
    assert counts, "expected a non-empty grid"
    return int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])


def _normalize(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = list(cells)
    min_r = min(r for r, _c in pts)
    min_c = min(c for _r, c in pts)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in pts))


def _bbox(cells: tuple[tuple[int, int], ...] | set[tuple[int, int]]) -> tuple[int, int]:
    pts = list(cells)
    return max(r for r, _c in pts) + 1, max(c for _r, c in pts) + 1


def _rotate(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, _w = _bbox(cells)
    return _normalize({(c, h - 1 - r) for r, c in cells})


def _flip_lr(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _h, w = _bbox(cells)
    return _normalize({(r, w - 1 - c) for r, c in cells})


def _shape_variants() -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[int, int], ...]] = []
    for base in _BASE_SHAPES:
        cur = _normalize(base)
        for _ in range(4):
            for cand in (cur, _flip_lr(cur)):
                norm = _normalize(cand)
                if norm not in seen:
                    seen.add(norm)
                    out.append(norm)
            cur = _rotate(cur)
    return out


_SHAPE_VARIANTS = _shape_variants()


def _central_cells(inner: int) -> set[tuple[int, int]]:
    if inner % 2 == 1:
        mid = inner // 2
        return {(mid, mid)}
    hi = inner // 2
    lo = hi - 1
    return {(lo, lo), (lo, hi), (hi, lo), (hi, hi)}


def _dock_shape(shape: tuple[tuple[int, int], ...], inner: int, side: str) -> tuple[tuple[int, int], ...] | None:
    h, w = _bbox(shape)
    if h > inner or w > inner:
        return None
    if side == "top":
        base_r = inner - h
        base_c = (inner - w) // 2
    elif side == "bottom":
        base_r = 0
        base_c = (inner - w) // 2
    elif side == "left":
        base_r = (inner - h) // 2
        base_c = inner - w
    elif side == "right":
        base_r = (inner - h) // 2
        base_c = 0
    else:
        raise ValueError(side)
    return tuple(sorted((int(base_r + r), int(base_c + c)) for r, c in shape))


def _slot_bounds(top: int, left: int, inner: int, slot_r: int, slot_c: int) -> tuple[int, int, int, int]:
    r0 = top + slot_r * (inner + 1)
    c0 = left + slot_c * (inner + 1)
    r1 = r0 + inner + 1
    c1 = c0 + inner + 1
    return int(r0), int(c0), int(r1), int(c1)


def _draw_frame(grid: np.ndarray, top: int, left: int, rows: int, cols: int, inner: int, color: int) -> None:
    frame_h = rows * inner + rows + 1
    frame_w = cols * inner + cols + 1
    for rr in range(rows + 1):
        r = top + rr * (inner + 1)
        grid[r, left : left + frame_w] = int(color)
    for cc in range(cols + 1):
        c = left + cc * (inner + 1)
        grid[top : top + frame_h, c] = int(color)


def _available_sides(slot_r: int, slot_c: int, rows: int, cols: int) -> list[str]:
    sides = []
    if slot_r == 0:
        sides.append("top")
    if slot_r == rows - 1:
        sides.append("bottom")
    if slot_c == 0:
        sides.append("left")
    if slot_c == cols - 1:
        sides.append("right")
    return sides


def _choose_slot_piece(inner: int, side: str) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[int, int]]:
    center_cells = _central_cells(inner)
    candidates = []
    for shape in _SHAPE_VARIANTS:
        placed = _dock_shape(shape, inner, side)
        if placed is None:
            continue
        overlap = sorted(set(placed) & center_cells)
        if not overlap:
            continue
        candidates.append((shape, placed, random.choice(overlap)))
    assert candidates, "expected at least one placeable shape"
    return random.choice(candidates)


def _stamp(grid: np.ndarray, cells: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> None:
    for r, c in cells:
        grid[int(top + r), int(left + c)] = int(color)


def _clear_fragment_positions(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], color: int, bg: int) -> None:
    r0, c0, r1, c1 = frame_bbox
    for r, c in np.argwhere(grid == int(color)):
        rr = int(r)
        cc = int(c)
        if not (r0 <= rr <= r1 and c0 <= cc <= c1):
            grid[rr, cc] = int(bg)


def _shape_from_points(points: np.ndarray) -> tuple[tuple[int, int], ...]:
    assert len(points) > 0
    min_r = int(points[:, 0].min())
    min_c = int(points[:, 1].min())
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in points))


def _can_place_fragment(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int, bg: int) -> bool:
    h, w = _bbox(shape)
    H, W = grid.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False
    for r, c in shape:
        rr = int(top + r)
        cc = int(left + c)
        if int(grid[rr, cc]) != int(bg):
            return False
        r0 = max(0, rr - 1)
        r1 = min(H, rr + 2)
        c0 = max(0, cc - 1)
        c1 = min(W, cc + 2)
        if np.any(grid[r0:r1, c0:c1] != int(bg)):
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored fragment into the same-colored slot of the gray frame.

    Input structure:
    - The background is black.
    - A large gray rectangle on the left is subdivided by gray grid lines into equal square slots.
    - Each slot contains one unique colored marker pixel and has exactly one one-cell border gap on one side.
    - The right side contains one small 4-connected fragment for each marker color.

    Rule:
    - Match every fragment to the slot whose marker has the same color.
    - The slot's missing side gives the entry direction; the fragment keeps its shape and orientation.
    - Translate the fragment straight into the slot, center it on the perpendicular axis, and dock it against the inner wall opposite the opening.
    - Remove the original marker and clear the fragment from the right side.

    Key insight:
    - Color solves the fragment-to-slot pairing, while the slot opening determines the only valid docking direction and wall-aligned placement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    non_bg = [int(v) for v in np.unique(g) if int(v) != int(bg)]
    assert non_bg, "expected non-background structure"
    counts = Counter(int(v) for v in g[g != int(bg)])
    frame_color = int(max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0])

    frame_mask = g == int(frame_color)
    frame_points = np.argwhere(frame_mask)
    assert len(frame_points) > 0, "expected a frame"
    fr0 = int(frame_points[:, 0].min())
    fc0 = int(frame_points[:, 1].min())
    fr1 = int(frame_points[:, 0].max())
    fc1 = int(frame_points[:, 1].max())

    frame_box = frame_mask[fr0 : fr1 + 1, fc0 : fc1 + 1]
    row_counts = frame_box.sum(axis=1)
    col_counts = frame_box.sum(axis=0)
    row_lines = [fr0 + idx for idx, cnt in enumerate(row_counts) if int(cnt) > frame_box.shape[1] // 2]
    col_lines = [fc0 + idx for idx, cnt in enumerate(col_counts) if int(cnt) > frame_box.shape[0] // 2]
    assert len(row_lines) >= 2 and len(col_lines) >= 2, "expected a partitioned frame"

    inner_heights = {int(row_lines[i + 1] - row_lines[i] - 1) for i in range(len(row_lines) - 1)}
    inner_widths = {int(col_lines[i + 1] - col_lines[i] - 1) for i in range(len(col_lines) - 1)}
    assert len(inner_heights) == 1 and len(inner_widths) == 1, "expected equal slot sizes"
    inner_h = inner_heights.pop()
    inner_w = inner_widths.pop()
    assert inner_h == inner_w and inner_h > 0, "expected equal square slots"
    inner = int(inner_h)

    out = g.copy()
    frame_bbox = (fr0, fc0, fr1, fc1)
    frame_area = np.zeros_like(g, dtype=bool)
    frame_area[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

    for r_idx in range(len(row_lines) - 1):
        for c_idx in range(len(col_lines) - 1):
            r0 = int(row_lines[r_idx])
            r1 = int(row_lines[r_idx + 1])
            c0 = int(col_lines[c_idx])
            c1 = int(col_lines[c_idx + 1])
            slot = g[r0 + 1 : r1, c0 + 1 : c1]
            marker_pts = np.argwhere((slot != int(bg)) & (slot != int(frame_color)))
            assert len(marker_pts) == 1, "expected one marker per slot"
            mr, mc = map(int, marker_pts[0])
            color = int(slot[mr, mc])

            side_flags = []
            if np.any(g[r0, c0 + 1 : c1] != int(frame_color)):
                side_flags.append("top")
            if np.any(g[r1, c0 + 1 : c1] != int(frame_color)):
                side_flags.append("bottom")
            if np.any(g[r0 + 1 : r1, c0] != int(frame_color)):
                side_flags.append("left")
            if np.any(g[r0 + 1 : r1, c1] != int(frame_color)):
                side_flags.append("right")
            assert len(side_flags) == 1, "expected one opening per slot"
            side = side_flags[0]

            fragment_pts = np.argwhere((g == int(color)) & (~frame_area))
            assert len(fragment_pts) >= 2, "expected one external fragment"
            shape = _shape_from_points(fragment_pts)
            placed = _dock_shape(shape, inner, side)
            assert placed is not None, "fragment must fit into its slot"

            out[r0 + 1 + mr, c0 + 1 + mc] = int(bg)
            _clear_fragment_positions(out, frame_bbox, color, bg)
            _stamp(out, placed, r0 + 1, c0 + 1, color)

    return out


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(_LAYOUTS)
    inner = 3 if cols == 3 else random.choice([3, 4])
    slot_count = rows * cols
    colors = random.sample(_PALETTE, slot_count)

    top = random.randint(1, 2)
    left = 1
    gap = random.randint(2, 3)
    right_margin = 1
    bottom_margin = random.randint(1, 2)

    frame_h = rows * inner + rows + 1
    frame_w = cols * inner + cols + 1

    scatter_min = 8 if slot_count == 4 else 10
    scatter_max = 30 - left - frame_w - gap - right_margin
    if scatter_max < scatter_min:
        return None
    scatter_w = random.randint(scatter_min, scatter_max)

    height_floor = 12 if slot_count == 4 else 14
    H_min = max(frame_h + top + bottom_margin, height_floor)
    H_max = min(30, H_min + 4)
    if H_min > H_max:
        return None
    H = random.randint(H_min, H_max)
    W = left + frame_w + gap + scatter_w + right_margin
    if max(H, W) > 30:
        return None

    inp = np.full((H, W), _BG, dtype=int)
    out = np.full((H, W), _BG, dtype=int)
    _draw_frame(inp, top, left, rows, cols, inner, _FRAME)
    _draw_frame(out, top, left, rows, cols, inner, _FRAME)

    slot_specs = []
    for idx, color in enumerate(colors):
        slot_r = idx // cols
        slot_c = idx % cols
        r0, c0, r1, c1 = _slot_bounds(top, left, inner, slot_r, slot_c)
        side = random.choice(_available_sides(slot_r, slot_c, rows, cols))
        shape, placed, marker_local = _choose_slot_piece(inner, side)
        mr, mc = marker_local

        if side == "top":
            inp[r0, c0 + 1 + mc] = _BG
            out[r0, c0 + 1 + mc] = _BG
        elif side == "bottom":
            inp[r1, c0 + 1 + mc] = _BG
            out[r1, c0 + 1 + mc] = _BG
        elif side == "left":
            inp[r0 + 1 + mr, c0] = _BG
            out[r0 + 1 + mr, c0] = _BG
        else:
            inp[r0 + 1 + mr, c1] = _BG
            out[r0 + 1 + mr, c1] = _BG

        inp[r0 + 1 + mr, c0 + 1 + mc] = int(color)
        _stamp(out, placed, r0 + 1, c0 + 1, color)

        slot_specs.append(
            {
                "color": int(color),
                "shape": shape,
            }
        )

    scatter_c0 = left + frame_w + gap
    order = list(slot_specs)
    random.shuffle(order)
    order.sort(key=lambda spec: (-len(spec["shape"]), spec["color"]))
    for spec in order:
        shape = spec["shape"]
        h, w = _bbox(shape)
        positions = [(r, c) for r in range(H - h + 1) for c in range(scatter_c0, W - w + 1)]
        random.shuffle(positions)
        placed = False
        for rr, cc in positions:
            if _can_place_fragment(inp, shape, rr, cc, _BG):
                _stamp(inp, shape, rr, cc, spec["color"])
                placed = True
                break
        if not placed:
            return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC fragment-slot puzzle")
