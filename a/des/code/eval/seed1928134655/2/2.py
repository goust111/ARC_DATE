from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()
_SIDES = ("top", "bottom", "left", "right")
_AREA_SHAPES = {
    1: [(1, 1)],
    3: [(1, 3), (3, 1)],
    9: [(3, 3)],
}
_MAX_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _extract_objects(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int] | str]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _components(g[1:-1, 1:-1] != 0)
    objects: list[dict[str, int | tuple[int, int, int, int] | str]] = []
    for comp in comps:
        rows = [r + 1 for r, _c in comp]
        cols = [c + 1 for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        color = int(g[rows[0], cols[0]])
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color)
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        assert hh % 2 == 1 and ww % 2 == 1
        touches = []
        if r0 == 1:
            touches.append("top")
        if r1 == h - 2:
            touches.append("bottom")
        if c0 == 1:
            touches.append("left")
        if c1 == w - 2:
            touches.append("right")
        assert len(touches) == 1
        objects.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "h": hh,
                "w": ww,
                "area": hh * ww,
                "side": touches[0],
            }
        )
    return objects


def _label_color(grid: np.ndarray, obj: dict[str, int | tuple[int, int, int, int] | str]) -> int:
    g = np.asarray(grid, dtype=int)
    side = str(obj["side"])
    r0, c0, r1, c1 = int(obj["r0"]), int(obj["c0"]), int(obj["r1"]), int(obj["c1"])
    if side == "top":
        vals = np.unique(g[0, c0 : c1 + 1])
    elif side == "bottom":
        vals = np.unique(g[g.shape[0] - 1, c0 : c1 + 1])
    elif side == "left":
        vals = np.unique(g[r0 : r1 + 1, 0])
    else:
        vals = np.unique(g[r0 : r1 + 1, g.shape[1] - 1])
    assert len(vals) == 1
    return int(vals[0])


def _target_rows(grid: np.ndarray, label: int) -> list[int]:
    g = np.asarray(grid, dtype=int)
    rows = {
        r
        for r in range(1, g.shape[0] - 1)
        if int(g[r, 0]) == label or int(g[r, g.shape[1] - 1]) == label
    }
    out = sorted(rows)
    assert out
    return out


def _target_cols(grid: np.ndarray, label: int) -> list[int]:
    g = np.asarray(grid, dtype=int)
    cols = {
        c
        for c in range(1, g.shape[1] - 1)
        if int(g[0, c]) == label or int(g[g.shape[0] - 1, c]) == label
    }
    out = sorted(cols)
    assert out
    return out


def _stamp(out: np.ndarray, color: int, shape: tuple[int, int], center_r: int, center_c: int) -> None:
    hh, ww = shape
    r0 = int(center_r - hh // 2)
    c0 = int(center_c - ww // 2)
    r1 = r0 + hh - 1
    c1 = c0 + ww - 1
    assert 1 <= r0 <= r1 < out.shape[0] - 1
    assert 1 <= c0 <= c1 < out.shape[1] - 1
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each edge-touching rectangle to the Cartesian product of border positions
    encoded by the color of the segment it touches.

    - Rows come from all left/right border cells of that label color.
    - Columns come from all top/bottom border cells of that label color.
    - Objects are drawn from small area to large area, so larger objects cover smaller ones.
    - The border is preserved unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    out[0, :] = g[0, :]
    out[-1, :] = g[-1, :]
    out[:, 0] = g[:, 0]
    out[:, -1] = g[:, -1]

    objects = _extract_objects(g)
    objects.sort(key=lambda obj: (int(obj["area"]), int(obj["r0"]), int(obj["c0"])))

    for obj in objects:
        label = _label_color(g, obj)
        rows = _target_rows(g, label)
        cols = _target_cols(g, label)
        shape = (int(obj["h"]), int(obj["w"]))
        for r in rows:
            for c in cols:
                _stamp(out, int(obj["color"]), shape, int(r), int(c))
    return out


def _interval_bounds(size: int, length: int) -> tuple[int, int]:
    lo = 1 + size // 2
    hi = length - size // 2
    return int(lo), int(hi)


def _pick_shape(area: int) -> tuple[int, int]:
    choices = _AREA_SHAPES[int(area)]
    return tuple(map(int, choices[int(_RNG.integers(0, len(choices)))]))


def _place_interval(side: np.ndarray, color: int, lo: int, hi: int, seg_len: int) -> tuple[int, int] | None:
    max_start = int(hi - seg_len + 1)
    starts = [
        start
        for start in range(int(lo), max_start + 1)
        if start - 1 + seg_len <= len(side) and np.all(side[start - 1 : start - 1 + seg_len] == -1)
    ]
    if not starts:
        return None
    start = int(starts[int(_RNG.integers(0, len(starts)))])
    side[start - 1 : start - 1 + seg_len] = int(color)
    return start, int(start + seg_len - 1)


def _interval_points(intervals: list[tuple[int, int]]) -> list[int]:
    pts = set()
    for start, end in intervals:
        pts.update(range(int(start), int(end) + 1))
    out = sorted(pts)
    assert out
    return out


def _choose_base_lengths(side: str, shape: tuple[int, int]) -> tuple[int, int]:
    hh, ww = shape
    row_len = int(_RNG.integers(1, 4))
    col_len = int(_RNG.integers(1, 4))
    if side in ("left", "right"):
        row_len = max(row_len, hh)
    if side in ("top", "bottom"):
        col_len = max(col_len, ww)
    return row_len, col_len


def _touch_center(
    side: str, shape: tuple[int, int], interval: tuple[int, int], height: int, width: int
) -> tuple[int, int] | None:
    hh, ww = shape
    start, end = interval
    if side == "top":
        row = 1 + hh // 2
        lo = max(int(start + ww // 2), 2 + ww // 2)
        hi = min(int(end - ww // 2), width - 3 - ww // 2)
        choices = list(range(lo, hi + 1))
        if not choices:
            return None
        return int(row), int(choices[int(_RNG.integers(0, len(choices)))])
    if side == "bottom":
        row = height - 2 - hh // 2
        lo = max(int(start + ww // 2), 2 + ww // 2)
        hi = min(int(end - ww // 2), width - 3 - ww // 2)
        choices = list(range(lo, hi + 1))
        if not choices:
            return None
        return int(row), int(choices[int(_RNG.integers(0, len(choices)))])
    if side == "left":
        col = 1 + ww // 2
        lo = max(int(start + hh // 2), 2 + hh // 2)
        hi = min(int(end - hh // 2), height - 3 - hh // 2)
        choices = list(range(lo, hi + 1))
        if not choices:
            return None
        return int(choices[int(_RNG.integers(0, len(choices)))]), int(col)
    col = width - 2 - ww // 2
    lo = max(int(start + hh // 2), 2 + hh // 2)
    hi = min(int(end - hh // 2), height - 3 - hh // 2)
    choices = list(range(lo, hi + 1))
    if not choices:
        return None
    return int(choices[int(_RNG.integers(0, len(choices)))]), int(col)


def _bbox_from_center(shape: tuple[int, int], center: tuple[int, int]) -> tuple[int, int, int, int]:
    hh, ww = shape
    rr, cc = center
    r0 = int(rr - hh // 2)
    c0 = int(cc - ww // 2)
    return r0, c0, int(r0 + hh - 1), int(c0 + ww - 1)


def _separated(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _fill_side(side: np.ndarray, filler_colors: np.ndarray) -> np.ndarray:
    filled = side.copy()
    for idx in range(len(filled)):
        if filled[idx] == -1:
            filled[idx] = int(filler_colors[int(_RNG.integers(0, len(filler_colors)))])
    assert np.all(filled > 0)
    return filled


def _paint_object_mask(shape: tuple[int, int], rows: list[int], cols: list[int], canvas_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(canvas_shape, dtype=bool)
    hh, ww = shape
    for rr in rows:
        for cc in cols:
            r0 = int(rr - hh // 2)
            c0 = int(cc - ww // 2)
            mask[r0 : r0 + hh, c0 : c0 + ww] = True
    return mask


def _build_object_specs(height: int, width: int, n_objects: int) -> list[dict] | None:
    colors = _RNG.permutation(np.arange(1, 10))
    label_colors = [int(v) for v in colors[:n_objects]]
    object_colors = [int(v) for v in colors[n_objects : 2 * n_objects]]
    filler_colors = colors[2 * n_objects :]
    areas = [1, 3, 9]
    chosen_areas = [int(areas[i]) for i in _RNG.choice(len(areas), size=n_objects, replace=False)]
    shapes = [_pick_shape(area) for area in chosen_areas]
    sides = [str(_SIDES[int(_RNG.integers(0, len(_SIDES)))]) for _ in range(n_objects)]

    side_arrays = {
        "top": np.full(width - 2, -1, dtype=int),
        "bottom": np.full(width - 2, -1, dtype=int),
        "left": np.full(height - 2, -1, dtype=int),
        "right": np.full(height - 2, -1, dtype=int),
    }

    order = sorted(range(n_objects), key=lambda i: max(shapes[i]), reverse=True)
    raw_specs: list[dict | None] = [None] * n_objects

    for idx in order:
        shape = shapes[idx]
        side = sides[idx]
        hh, ww = shape
        row_lo, row_hi = _interval_bounds(hh, height - 2)
        col_lo, col_hi = _interval_bounds(ww, width - 2)
        row_len, col_len = _choose_base_lengths(side, shape)

        if side in ("top", "bottom"):
            horiz_side = side
            vert_side = "left" if int(_RNG.integers(0, 2)) == 0 else "right"
        else:
            vert_side = side
            horiz_side = "top" if int(_RNG.integers(0, 2)) == 0 else "bottom"

        vert_intervals: list[tuple[int, int]] = []
        horiz_intervals: list[tuple[int, int]] = []

        base_vert = _place_interval(side_arrays[vert_side], label_colors[idx], row_lo, row_hi, row_len)
        if base_vert is None:
            return None
        vert_intervals.append(base_vert)

        base_horiz = _place_interval(side_arrays[horiz_side], label_colors[idx], col_lo, col_hi, col_len)
        if base_horiz is None:
            return None
        horiz_intervals.append(base_horiz)

        if int(_RNG.integers(0, 2)) == 1:
            extra_side = "right" if vert_side == "left" else "left"
            extra_len = int(_RNG.integers(1, 3))
            extra_vert = _place_interval(side_arrays[extra_side], label_colors[idx], row_lo, row_hi, extra_len)
            if extra_vert is not None:
                vert_intervals.append(extra_vert)

        if int(_RNG.integers(0, 2)) == 1:
            extra_side = "bottom" if horiz_side == "top" else "top"
            extra_len = int(_RNG.integers(1, 3))
            extra_horiz = _place_interval(side_arrays[extra_side], label_colors[idx], col_lo, col_hi, extra_len)
            if extra_horiz is not None:
                horiz_intervals.append(extra_horiz)

        rows = _interval_points(vert_intervals)
        cols = _interval_points(horiz_intervals)
        if len(rows) * len(cols) > 9:
            return None

        touch_interval = base_horiz if side in ("top", "bottom") else base_vert
        center = _touch_center(side, shape, touch_interval, height, width)
        if center is None:
            return None
        bbox = _bbox_from_center(shape, center)
        if not (1 <= bbox[0] <= bbox[2] < height - 1 and 1 <= bbox[1] <= bbox[3] < width - 1):
            return None

        raw_specs[idx] = {
            "label": label_colors[idx],
            "color": object_colors[idx],
            "shape": shape,
            "side": side,
            "rows": rows,
            "cols": cols,
            "bbox": bbox,
        }

    specs = [spec for spec in raw_specs if spec is not None]
    assert len(specs) == n_objects

    boxes = [tuple(map(int, spec["bbox"])) for spec in specs]
    for i, box_a in enumerate(boxes):
        for box_b in boxes[i + 1 :]:
            if not _separated(box_a, box_b):
                return None

    top = _fill_side(side_arrays["top"], filler_colors)
    bottom = _fill_side(side_arrays["bottom"], filler_colors)
    left = _fill_side(side_arrays["left"], filler_colors)
    right = _fill_side(side_arrays["right"], filler_colors)

    return [
        {
            **spec,
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
        }
        for spec in specs
    ]


def _render_input(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    top = np.asarray(specs[0]["top"], dtype=int)
    bottom = np.asarray(specs[0]["bottom"], dtype=int)
    left = np.asarray(specs[0]["left"], dtype=int)
    right = np.asarray(specs[0]["right"], dtype=int)
    grid[0, 1:-1] = top
    grid[-1, 1:-1] = bottom
    grid[1:-1, 0] = left
    grid[1:-1, -1] = right
    grid[0, 0] = int(top[0])
    grid[0, -1] = int(top[-1])
    grid[-1, 0] = int(bottom[0])
    grid[-1, -1] = int(bottom[-1])

    for spec in specs:
        r0, c0, r1, c1 = map(int, spec["bbox"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])
    return grid


def _pairwise_overlap(specs: list[dict], canvas_shape: tuple[int, int]) -> bool:
    masks = [
        _paint_object_mask(tuple(spec["shape"]), list(spec["rows"]), list(spec["cols"]), canvas_shape)
        for spec in specs
    ]
    for i, mask_a in enumerate(masks):
        for mask_b in masks[i + 1 :]:
            if bool((mask_a & mask_b).any()):
                return True
    return False


def _attempt_generate() -> np.ndarray | None:
    n_objects = int(_RNG.integers(2, 4))
    height = int(_RNG.integers(14, 20))
    width = int(_RNG.integers(14, 20))
    want_overlap = bool(int(_RNG.integers(0, 2)))

    specs = _build_object_specs(height, width, n_objects)
    if specs is None:
        return None

    if not any(len(spec["rows"]) > 1 or len(spec["cols"]) > 1 for spec in specs):
        return None

    has_overlap = _pairwise_overlap(specs, (height, width))
    if has_overlap != want_overlap:
        return None

    grid = _render_input(height, width, specs)
    objects = _extract_objects(grid)
    if len(objects) != n_objects:
        return None

    out = main(grid)
    interior = out[1:-1, 1:-1]
    density = float(np.count_nonzero(interior) / interior.size)
    if not (0.05 <= density <= 0.65):
        return None
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[0, :], grid[0, :]):
        return None
    if not np.array_equal(out[-1, :], grid[-1, :]):
        return None
    if not np.array_equal(out[:, 0], grid[:, 0]):
        return None
    if not np.array_equal(out[:, -1], grid[:, -1]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
