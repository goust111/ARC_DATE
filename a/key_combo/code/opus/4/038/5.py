from __future__ import annotations

import numpy as np


BACKGROUND = 0
RECT_COLOR = 8
CONTROL_COLOR = 9
INTERIOR_COLOR = 4
QUADRANT_COLORS = {
    (True, True): 6,    # upper-left
    (True, False): 1,   # upper-right
    (False, True): 2,   # lower-left
    (False, False): 3,  # lower-right
}
SIDES = ("top", "bottom", "left", "right")

_RNG = np.random.default_rng()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = {int(v) for v in np.unique(g)}
    assert colors <= {BACKGROUND, RECT_COLOR, CONTROL_COLOR}

    rect_mask = g == RECT_COLOR
    rect_bbox = _bbox(rect_mask)
    assert rect_bbox is not None, "expected one cyan rectangle"
    r0, c0, r1, c1 = rect_bbox
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    assert rect_h >= 4 and rect_w >= 4
    assert int(rect_mask.sum()) == rect_h * rect_w
    assert np.all(rect_mask[r0 : r1 + 1, c0 : c1 + 1]), "cyan area must be a solid rectangle"

    controls: list[tuple[int, int, str]] = []
    for r_raw, c_raw in np.argwhere(g == CONTROL_COLOR):
        r = int(r_raw)
        c = int(c_raw)
        side_hits = []
        if r < r0 and c0 <= c <= c1:
            side_hits.append("top")
        if r > r1 and c0 <= c <= c1:
            side_hits.append("bottom")
        if c < c0 and r0 <= r <= r1:
            side_hits.append("left")
        if c > c1 and r0 <= r <= r1:
            side_hits.append("right")
        assert len(side_hits) == 1, "every control pixel must lie on exactly one rectangle side extension line"
        controls.append((r, c, side_hits[0]))

    return {
        "shape": tuple(map(int, g.shape)),
        "bbox": rect_bbox,
        "controls": tuple(controls),
    }


def _blocked_lines(spec: dict) -> dict[str, set[int]]:
    blocked = {side: set() for side in SIDES}
    for r, c, side in spec["controls"]:
        if side in ("top", "bottom"):
            blocked[side].add(int(c))
        else:
            blocked[side].add(int(r))
    return blocked


def _shape_mask_from_spec(spec: dict, *, ignore_controls: bool = False) -> np.ndarray:
    h, w = spec["shape"]
    r0, c0, r1, c1 = spec["bbox"]
    blocked = {side: set() for side in SIDES} if ignore_controls else _blocked_lines(spec)

    shape_mask = np.zeros((h, w), dtype=bool)
    shape_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    for c in range(c0, c1 + 1):
        if c not in blocked["top"]:
            shape_mask[: r0 + 1, c] = True
        if c not in blocked["bottom"]:
            shape_mask[r1:h, c] = True

    for r in range(r0, r1 + 1):
        if r not in blocked["left"]:
            shape_mask[r, : c0 + 1] = True
        if r not in blocked["right"]:
            shape_mask[r, c1:w] = True

    return shape_mask


def _border_mask(shape_mask: np.ndarray) -> np.ndarray:
    border = np.zeros_like(shape_mask, dtype=bool)
    for r_raw, c_raw in np.argwhere(shape_mask):
        r = int(r_raw)
        c = int(c_raw)
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(shape_mask.shape, nr, nc) or not shape_mask[nr, nc]:
                border[r, c] = True
                break
    return border


def _paint_shape(
    shape_mask: np.ndarray,
    controls: tuple[tuple[int, int, str], ...],
    center: tuple[int, int],
    *,
    recolor_interior: bool = True,
) -> np.ndarray:
    center_r, center_c = center
    border = _border_mask(shape_mask)
    out = np.full(shape_mask.shape, BACKGROUND, dtype=int)

    for r_raw, c_raw in np.argwhere(shape_mask):
        r = int(r_raw)
        c = int(c_raw)
        if recolor_interior and not border[r, c]:
            out[r, c] = INTERIOR_COLOR
            continue
        out[r, c] = QUADRANT_COLORS[(r < center_r, c < center_c)]

    for r, c, _side in controls:
        out[int(r), int(c)] = CONTROL_COLOR
    return out


def _expected_output(spec: dict) -> np.ndarray:
    shape_mask = _shape_mask_from_spec(spec)
    top, left, bottom, right = _bbox(shape_mask)
    center = ((top + bottom) // 2, (left + right) // 2)
    return _paint_shape(shape_mask, spec["controls"], center, recolor_interior=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend an inset cyan rectangle along unblocked rows and columns, then recolor the resulting outline by quadrant while keeping interior yellow.

    Input structure:
    - The grid has a black background, one solid cyan rectangle, and some maroon control pixels outside it.
    - Each maroon control lies in exactly one side strip: above or below the rectangle in one of its columns, or left or right of the rectangle in one of its rows.
    - Control pixels indicate which rectangle columns or rows must not extend outward on that side.

    Rule:
    - Keep the original cyan rectangle, then extend every unblocked rectangle column upward and downward to the grid edge, and every unblocked rectangle row leftward and rightward to the grid edge.
    - Compute the bounding-box center of the full extended shape. Border pixels are shape pixels with at least one 4-neighbor outside the shape; all other shape pixels are interior.
    - Recolor border pixels by quadrant relative to that center: upper-left magenta, upper-right blue, lower-left red, lower-right green; recolor every interior pixel yellow; finally place the maroon control pixels back on top unchanged.

    Key insight:
    - The control pixels do not mark output colors directly; they carve holes out of the rectangle's four directional stripe extensions, and the irregular extended silhouette is what determines both the border and the quadrant-based recoloring.
    """
    spec = _parse_grid(np.asarray(grid, dtype=int))
    return _expected_output(spec)


def _side_count_options(rect_h: int, rect_w: int) -> list[tuple[int, int, int, int]]:
    top_max = min(3, rect_w - 1)
    bottom_max = min(3, rect_w - 1)
    left_max = min(3, rect_h - 1)
    right_max = min(3, rect_h - 1)
    options = []
    for top in range(top_max + 1):
        for bottom in range(bottom_max + 1):
            for left in range(left_max + 1):
                for right in range(right_max + 1):
                    counts = (top, bottom, left, right)
                    total = sum(counts)
                    if total == 0:
                        continue
                    if 0 not in counts:
                        continue
                    options.append(counts)
    return options


def _choose_count_pattern(rect_h: int, rect_w: int) -> tuple[int, int, int, int]:
    options = _side_count_options(rect_h, rect_w)
    scored = []
    for counts in options:
        top, bottom, left, right = counts
        vertical = int(top > 0 or bottom > 0)
        horizontal = int(left > 0 or right > 0)
        zeros = sum(count == 0 for count in counts)
        score = 1
        if vertical and horizontal:
            score += 6
        if zeros == 1:
            score += 4
        if max(counts) >= 2:
            score += 2
        if min(top + bottom, left + right) > 0:
            score += 1
        scored.append((counts, score))

    weights = np.array([score for _, score in scored], dtype=float)
    weights /= weights.sum()
    choice = int(_RNG.choice(len(scored), p=weights))
    return scored[choice][0]


def _pick_distinct(values: np.ndarray, count: int) -> np.ndarray:
    return np.sort(_RNG.choice(values, size=count, replace=False))


def _biased_start(limit: int, length: int, center: int) -> int:
    lo = 1
    hi = limit - length - 1
    assert lo <= hi
    if hi == lo:
        return lo

    mode = int(_RNG.integers(0, 4))
    if mode == 0:
        return int(_RNG.integers(lo, hi + 1))
    if mode == 1:
        target = center - length // 2
        return int(np.clip(target + int(_RNG.integers(-2, 3)), lo, hi))
    if mode == 2:
        return int(_RNG.integers(lo, min(hi, lo + 4) + 1))
    return int(_RNG.integers(max(lo, hi - 4), hi + 1))


def _render_input(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    controls: tuple[tuple[int, int, str], ...],
) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = RECT_COLOR
    for r, c, _side in controls:
        grid[int(r), int(c)] = CONTROL_COLOR
    return grid


def _make_control_coords(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    counts: tuple[int, int, int, int],
) -> tuple[tuple[int, int, str], ...]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    rect_rows = np.arange(r0, r1 + 1)
    rect_cols = np.arange(c0, c1 + 1)
    controls: list[tuple[int, int, str]] = []

    top_count, bottom_count, left_count, right_count = counts
    if top_count:
        top_cols = _pick_distinct(rect_cols, top_count)
        for c in top_cols:
            controls.append((int(_RNG.integers(0, r0)), int(c), "top"))
    if bottom_count:
        bottom_cols = _pick_distinct(rect_cols, bottom_count)
        for c in bottom_cols:
            controls.append((int(_RNG.integers(r1 + 1, h)), int(c), "bottom"))
    if left_count:
        left_rows = _pick_distinct(rect_rows, left_count)
        for r in left_rows:
            controls.append((int(r), int(_RNG.integers(0, c0)), "left"))
    if right_count:
        right_rows = _pick_distinct(rect_rows, right_count)
        for r in right_rows:
            controls.append((int(r), int(_RNG.integers(c1 + 1, w)), "right"))

    return tuple(controls)


def _candidate_spec() -> dict:
    h = int(_RNG.integers(18, 30))
    w = int(_RNG.integers(18, 30))
    rect_h = int(_RNG.integers(5, min(10, h - 1)))
    rect_w = int(_RNG.integers(5, min(11, w - 1)))

    center_r = h // 2
    center_c = w // 2
    r0 = _biased_start(h, rect_h, center_r)
    c0 = _biased_start(w, rect_w, center_c)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    counts = _choose_count_pattern(rect_h, rect_w)
    controls = _make_control_coords((h, w), (r0, c0, r1, c1), counts)
    return {
        "shape": (h, w),
        "bbox": (r0, c0, r1, c1),
        "controls": controls,
    }


def _validate_spec(spec: dict) -> bool:
    h, w = spec["shape"]
    r0, c0, r1, c1 = spec["bbox"]
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1

    if not (h <= 30 and w <= 30):
        return False
    if min(r0, c0, h - 1 - r1, w - 1 - c1) < 1:
        return False

    by_side = {side: [] for side in SIDES}
    for r, c, side in spec["controls"]:
        by_side[side].append((r, c))
        if side == "top" and not (0 <= r < r0 and c0 <= c <= c1):
            return False
        if side == "bottom" and not (r1 < r < h and c0 <= c <= c1):
            return False
        if side == "left" and not (r0 <= r <= r1 and 0 <= c < c0):
            return False
        if side == "right" and not (r0 <= r <= r1 and c1 < c < w):
            return False

    blocked = _blocked_lines(spec)
    if any(len(blocked[side]) != len(by_side[side]) for side in SIDES):
        return False
    if len(blocked["top"]) > rect_w - 1 or len(blocked["bottom"]) > rect_w - 1:
        return False
    if len(blocked["left"]) > rect_h - 1 or len(blocked["right"]) > rect_h - 1:
        return False
    if not any(len(by_side[side]) == 0 for side in SIDES):
        return False
    if sum(len(by_side[side]) for side in SIDES) == 0:
        return False

    grid = _render_input(spec["shape"], spec["bbox"], spec["controls"])
    if not np.array_equal(main(grid), _expected_output(spec)):
        return False

    shape_mask = _shape_mask_from_spec(spec)
    if int(shape_mask.sum()) <= rect_h * rect_w:
        return False

    actual = _expected_output(spec)

    no_control_mask = _shape_mask_from_spec(spec, ignore_controls=True)
    no_control_out = _paint_shape(
        no_control_mask,
        spec["controls"],
        ((_bbox(no_control_mask)[0] + _bbox(no_control_mask)[2]) // 2, (_bbox(no_control_mask)[1] + _bbox(no_control_mask)[3]) // 2),
    )
    if np.array_equal(actual, no_control_out):
        return False

    rect_center_out = _paint_shape(
        shape_mask,
        spec["controls"],
        ((r0 + r1) // 2, (c0 + c1) // 2),
    )
    if np.array_equal(actual, rect_center_out):
        return False

    top, left, bottom, right = _bbox(shape_mask)
    center = ((top + bottom) // 2, (left + right) // 2)
    no_interior_out = _paint_shape(shape_mask, spec["controls"], center, recolor_interior=False)
    if np.array_equal(actual, no_interior_out):
        return False

    border_colors = {int(v) for v in np.unique(actual)} - {BACKGROUND, CONTROL_COLOR, INTERIOR_COLOR}
    if len(border_colors) < 3:
        return False

    if actual.shape != (h, w):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        spec = _candidate_spec()
        if _validate_spec(spec):
            return _render_input(spec["shape"], spec["bbox"], spec["controls"])
    raise RuntimeError("failed to generate a valid puzzle instance")
