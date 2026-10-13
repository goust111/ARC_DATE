from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_BASE_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
]


def _normalize(coords: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    pts = sorted({(int(r), int(c)) for r, c in coords})
    r0 = min(r for r, _c in pts)
    c0 = min(c for _r, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _transform_shape(shape: tuple[tuple[int, int], ...], rot: int, flip: bool) -> tuple[tuple[int, int], ...]:
    pts = np.array(shape, dtype=int)
    for _ in range(int(rot) % 4):
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    if flip:
        pts = np.stack([pts[:, 0], -pts[:, 1]], axis=1)
    return _normalize(map(tuple, pts.tolist()))


def _random_shape() -> tuple[tuple[int, int], ...]:
    return _transform_shape(random.choice(_BASE_SHAPES), random.randrange(4), bool(random.getrandbits(1)))


def _shape_hw(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return max(rows) + 1, max(cols) + 1


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int, color: int) -> None:
    for dr, dc in shape:
        grid[int(r0 + dr), int(c0 + dc)] = int(color)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in _DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(sorted(coords), dtype=int))
    comps.sort(key=lambda pts: (int(pts[:, 0].min()), int(pts[:, 1].min())))
    return comps


def _barrier_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    cols = [c for c in range(w) if np.all(grid[:, c] == RED)]
    assert len(cols) == 1, "expected exactly one full-height red barrier"
    c = int(cols[0])
    assert int(np.count_nonzero(grid == RED)) == h, "barrier must occupy only one full column"
    return c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide side objects into the central red barrier, then project their travel distances as opposite-side horizontal step rays.

    Input structure:
    - The background is black.
    - A single red vertical barrier of width 1 spans the full grid height.
    - Blue 4-connected objects lie on the left of the barrier and green 4-connected objects lie on the right.

    Rule:
    - Keep the red barrier unchanged.
    - Slide each blue object horizontally right until its rightmost pixel becomes adjacent to the barrier; do the symmetric leftward slide for each green object.
    - For each slid blue object, draw blue horizontal rays on the opposite side, one per occupied row, starting next to the barrier with length equal to that object's slide distance; do the symmetric leftward rays for green objects.
    - If a ray reaches cells already occupied by a slid object, keep the object pixel and skip the ray there.

    Key insight:
    - The hidden numeric value is each object's slide distance to the barrier, and that distance is transferred across the barrier as the length of a row-wise stepped ray pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bar_c = _barrier_col(g)

    blue_comps = _find_components(g == BLUE)
    green_comps = _find_components(g == GREEN)
    assert 2 <= len(blue_comps) <= 4, "expected 2-4 blue objects"
    assert len(blue_comps) == len(green_comps), "left and right sides should have the same object count"

    out = np.full((h, w), BG, dtype=int)
    out[:, bar_c] = RED
    records: list[tuple[int, int, tuple[int, ...]]] = []

    for color, comps, direction in ((BLUE, blue_comps, 1), (GREEN, green_comps, -1)):
        for comp in comps:
            rows = tuple(sorted({int(r) for r, _c in comp.tolist()}))
            if color == BLUE:
                edge = int(comp[:, 1].max())
                distance = int(bar_c - edge - 1)
                shift = distance
            else:
                edge = int(comp[:, 1].min())
                distance = int(edge - bar_c - 1)
                shift = -distance
            assert distance >= 1, "objects must start away from the barrier"
            moved = comp.copy()
            moved[:, 1] += shift
            out[moved[:, 0], moved[:, 1]] = color
            records.append((color, distance, rows))

    for color, distance, rows in records:
        if color == BLUE:
            cols = range(bar_c + 1, bar_c + distance + 1)
        else:
            cols = range(bar_c - 1, bar_c - distance - 1, -1)
        for r in rows:
            for c in cols:
                if out[int(r), int(c)] == BG:
                    out[int(r), int(c)] = color

    return out


def _assign_unique_distances(max_distances: list[int], required: tuple[int, int] | None = None) -> list[int] | None:
    n = len(max_distances)
    allowed = [list(range(2, int(mx) + 1)) for mx in max_distances]
    if any(not vals for vals in allowed):
        return None

    order = list(range(n))
    if required is not None:
        req_idx, _req_min = required
        order.remove(req_idx)
        order.sort(key=lambda i: (len(allowed[i]), random.random()))
        order.insert(0, req_idx)
    else:
        order.sort(key=lambda i: (len(allowed[i]), random.random()))

    result = [0] * n

    def rec(k: int, used: set[int]) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        vals = allowed[idx][:]
        random.shuffle(vals)
        if required is not None and idx == required[0]:
            vals = [v for v in vals if v >= required[1]]
            if not vals:
                return False
        vals.sort(reverse=bool(random.getrandbits(1)))
        for val in vals:
            if val in used:
                continue
            result[idx] = int(val)
            used.add(int(val))
            if rec(k + 1, used):
                return True
            used.remove(int(val))
        return False

    return result if rec(0, set()) else None


def _expected_from_blueprint(
    height: int,
    width: int,
    barrier_col: int,
    left_specs: list[dict],
    right_specs: list[dict],
) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    out[:, barrier_col] = RED

    for spec in left_specs:
        h_shape, w_shape = _shape_hw(spec["shape"])
        final_c0 = barrier_col - w_shape
        _paint_shape(out, spec["shape"], spec["r0"], final_c0, BLUE)
    for spec in right_specs:
        final_c0 = barrier_col + 1
        _paint_shape(out, spec["shape"], spec["r0"], final_c0, GREEN)

    for spec in left_specs:
        rows = sorted({spec["r0"] + r for r, _c in spec["shape"]})
        for r in rows:
            for c in range(barrier_col + 1, barrier_col + spec["distance"] + 1):
                if out[r, c] == BG:
                    out[r, c] = BLUE
    for spec in right_specs:
        rows = sorted({spec["r0"] + r for r, _c in spec["shape"]})
        for r in rows:
            for c in range(barrier_col - 1, barrier_col - spec["distance"] - 1, -1):
                if out[r, c] == BG:
                    out[r, c] = GREEN
    return out


def _build_specs(side_count: int) -> tuple[list[dict], list[dict], int, int, int] | None:
    left_shapes = [{"shape": _random_shape()} for _ in range(side_count)]
    right_shapes = [{"shape": _random_shape()} for _ in range(side_count)]

    for spec in left_shapes + right_shapes:
        spec["h"], spec["w"] = _shape_hw(spec["shape"])

    band_count = min(6, side_count + random.randint(1, 2))
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    band_stride = 4
    band_height = 3
    height = top_margin + bottom_margin + band_count * band_height + (band_count - 1)
    width = random.randint(20, 26)
    barrier_col = width // 2

    left_bands = sorted(random.sample(range(band_count), side_count))
    right_bands = sorted(random.sample(range(band_count), side_count))
    force_occlusion = random.random() < 0.65
    if force_occlusion:
        chosen_band = random.choice(left_bands)
        if chosen_band not in right_bands:
            replace_idx = random.randrange(side_count)
            right_bands[replace_idx] = chosen_band
            right_bands = sorted(right_bands)

    for idx, spec in enumerate(left_shapes):
        band_top = top_margin + left_bands[idx] * band_stride
        spec["r0"] = band_top + random.randint(0, band_height - spec["h"])
    for idx, spec in enumerate(right_shapes):
        band_top = top_margin + right_bands[idx] * band_stride
        spec["r0"] = band_top + random.randint(0, band_height - spec["h"])

    overlap_pair: tuple[int, int] | None = None
    if force_occlusion:
        shared = [(i, j) for i, lb in enumerate(left_bands) for j, rb in enumerate(right_bands) if lb == rb]
        if not shared:
            return None
        overlap_pair = random.choice(shared)
        i, j = overlap_pair
        band_top = top_margin + left_bands[i] * band_stride
        left_shapes[i]["r0"] = band_top
        right_shapes[j]["r0"] = band_top

    left_caps = [min(6, barrier_col - spec["w"]) for spec in left_shapes]
    right_caps = [min(6, width - barrier_col - spec["w"] - 1) for spec in right_shapes]
    if min(left_caps) < 2 or min(right_caps) < 2:
        return None

    left_distances = _assign_unique_distances(left_caps)
    right_distances = _assign_unique_distances(right_caps)

    if force_occlusion and overlap_pair is not None:
        i, j = overlap_pair
        left_need = right_shapes[j]["w"] + 1
        right_need = left_shapes[i]["w"] + 1
        prefer_left = _assign_unique_distances(left_caps, required=(i, left_need))
        prefer_right = _assign_unique_distances(right_caps, required=(j, right_need))
        if prefer_left is not None:
            left_distances = prefer_left
        elif prefer_right is not None:
            right_distances = prefer_right
        else:
            return None

    if left_distances is None or right_distances is None:
        return None

    for spec, dist in zip(left_shapes, left_distances):
        spec["distance"] = int(dist)
    for spec, dist in zip(right_shapes, right_distances):
        spec["distance"] = int(dist)

    return left_shapes, right_shapes, height, width, barrier_col


def _gen_attempt() -> np.ndarray | None:
    side_count = random.randint(2, 4)
    built = _build_specs(side_count)
    if built is None:
        return None
    left_specs, right_specs, height, width, barrier_col = built

    inp = np.zeros((height, width), dtype=int)
    inp[:, barrier_col] = RED

    for spec in left_specs:
        start_c0 = barrier_col - spec["w"] - spec["distance"]
        if start_c0 < 0:
            return None
        _paint_shape(inp, spec["shape"], spec["r0"], start_c0, BLUE)
    for spec in right_specs:
        start_c0 = barrier_col + 1 + spec["distance"]
        if start_c0 + spec["w"] > width:
            return None
        _paint_shape(inp, spec["shape"], spec["r0"], start_c0, GREEN)

    blue_comps = _find_components(inp == BLUE)
    green_comps = _find_components(inp == GREEN)
    if len(blue_comps) != side_count or len(green_comps) != side_count:
        return None

    expected = _expected_from_blueprint(height, width, barrier_col, left_specs, right_specs)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    left_rows = [{spec["r0"] + r for r, _c in spec["shape"]} for spec in left_specs]
    right_rows = [{spec["r0"] + r for r, _c in spec["shape"]} for spec in right_specs]
    if not any(len(rows) >= 2 for rows in left_rows + right_rows):
        return None
    if len({spec["distance"] for spec in left_specs}) != side_count:
        return None
    if len({spec["distance"] for spec in right_specs}) != side_count:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
