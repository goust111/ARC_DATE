from __future__ import annotations

from collections import deque

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_component(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return (r0, c0, r1, c1), mask


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _is_cross_component(coords: np.ndarray) -> bool:
    if coords.shape[0] != 5:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return False
    mask = np.zeros((3, 3), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.array(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ],
        dtype=bool,
    )
    return bool(np.array_equal(mask, expected))


def _pair_rectangle_shape(comps: list[np.ndarray]) -> tuple[int, int] | None:
    if len(comps) != 2:
        return None
    crop_a = _crop_component(comps[0])[1]
    crop_b = _crop_component(comps[1])[1]
    if crop_a.shape != crop_b.shape:
        return None
    if np.any(crop_a & crop_b):
        return None
    if not np.all(crop_a | crop_b):
        return None
    return int(crop_a.shape[0]), int(crop_a.shape[1])


def _collect_color_components(grid: np.ndarray, bg: int) -> dict[int, list[np.ndarray]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    out: dict[int, list[np.ndarray]] = {}
    for color in colors:
        out[color] = _components_from_mask(grid == color, connectivity=8)
    return out


def _find_marker_color(components_by_color: dict[int, list[np.ndarray]]) -> int:
    colors = list(components_by_color)
    candidates: list[int] = []
    for color, comps in components_by_color.items():
        if not comps or not all(_is_cross_component(comp) for comp in comps):
            continue
        others = [other for other in colors if other != color]
        if len(comps) != len(others):
            continue
        if all(_pair_rectangle_shape(components_by_color[other]) is not None for other in others):
            candidates.append(int(color))
    assert len(candidates) == 1, "expected exactly one marker color"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments, overlay each pair into a solid rectangle, and
    move every rectangle to the matching grey cross in row-major order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    components_by_color = _collect_color_components(g, bg)
    marker_color = _find_marker_color(components_by_color)

    marker_entries = []
    for comp in components_by_color[marker_color]:
        r0, c0, _, _ = _bbox(comp)
        marker_entries.append((r0 + 1, c0 + 1))
    marker_entries.sort()

    pair_entries = []
    for color, comps in components_by_color.items():
        if color == marker_color:
            continue
        shape = _pair_rectangle_shape(comps)
        assert shape is not None, "each non-marker color must be exactly one complementary pair"
        boxes = [_crop_component(comp)[0] for comp in comps]
        first_box = min(boxes, key=lambda box: (box[0], box[1]))
        pair_entries.append((first_box[0], first_box[1], int(color), shape))
    pair_entries.sort(key=lambda item: (item[0], item[1]))

    assert len(pair_entries) == len(marker_entries), "markers and fragment pairs must match in count"

    out = np.full_like(g, bg)
    h, w = g.shape
    for (_, _, color, (rh, rw)), (cr, cc) in zip(pair_entries, marker_entries):
        top = int(cr - rh // 2)
        left = int(cc - rw // 2)
        assert 0 <= top and 0 <= left and top + rh <= h and left + rw <= w
        out[top : top + rh, left : left + rw] = int(color)
    return out


_RNG = np.random.default_rng()
_GEN_ATTEMPTS = 400
_PARTITION_ATTEMPTS = 200
_PLACEMENT_ATTEMPTS = 200


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[_randint(0, len(seq) - 1)]


def _shuffled(seq):
    items = list(seq)
    _RNG.shuffle(items)
    return items


def _sample(seq, k: int):
    idx = _RNG.choice(len(seq), size=int(k), replace=False)
    return [seq[int(i)] for i in idx]


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _mask_connected(mask: np.ndarray) -> bool:
    comps = _components_from_mask(mask, connectivity=8)
    return len(comps) == 1


def _frontier(region: set[tuple[int, int]], unassigned: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    cells = set()
    for r, c in region:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in unassigned:
                cells.add((nr, nc))
    return list(cells)


def _make_partition(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    rr, cc = np.indices((h, w))
    mask_a = ((rr + cc) % 2) == 0

    move_count = _randint(0, max(1, (h * w) // 6))
    for _ in range(move_count):
        for _ in range(40):
            r = _randint(0, h - 1)
            c = _randint(0, w - 1)
            trial = mask_a.copy()
            trial[r, c] = ~trial[r, c]
            trial_b = ~trial
            if not (_touches_all_sides(trial) and _touches_all_sides(trial_b)):
                continue
            if not (_mask_connected(trial) and _mask_connected(trial_b)):
                continue
            mask_a = trial
            break

    mask_b = ~mask_a
    if _RNG.random() < 0.5:
        return mask_a, mask_b
    return mask_b, mask_a


def _can_place(occ: np.ndarray, top: int, left: int, height: int, width: int, margin: int) -> bool:
    h, w = occ.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(h, top + height + margin)
    c1 = min(w, left + width + margin)
    return not bool(occ[r0:r1, c0:c1].any())


def _mark_occ(occ: np.ndarray, top: int, left: int, height: int, width: int, margin: int) -> None:
    h, w = occ.shape
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(h, top + height + margin)
    c1 = min(w, left + width + margin)
    occ[r0:r1, c0:c1] = True


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = int(color)


def _paint_cross(grid: np.ndarray, center: tuple[int, int], color: int) -> None:
    r, c = center
    grid[r, c] = color
    grid[r - 1, c] = color
    grid[r + 1, c] = color
    grid[r, c - 1] = color
    grid[r, c + 1] = color


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = _choice([2, 3, 4])
    if pair_count == 2:
        h_range = (22, 30)
        w_range = (22, 32)
        size_choices = [5, 7, 9, 11]
    elif pair_count == 3:
        h_range = (26, 34)
        w_range = (26, 34)
        size_choices = [5, 7, 9]
    else:
        h_range = (30, 36)
        w_range = (30, 36)
        size_choices = [5, 7, 9]

    h = _randint(*h_range)
    w = _randint(*w_range)

    colors = list(range(10))
    bg = _choice(colors)
    remaining = [c for c in colors if c != bg]
    marker_color = _choice(remaining)
    piece_colors = _sample([c for c in remaining if c != marker_color], pair_count)

    specs = []
    for color in piece_colors:
        rect_h = _choice(size_choices)
        rect_w = _choice(size_choices)
        if rect_h >= h - 2 or rect_w >= w - 2:
            return None
        try:
            mask_a, mask_b = _make_partition(rect_h, rect_w)
        except RuntimeError:
            return None
        specs.append(
            {
                "color": int(color),
                "shape": (rect_h, rect_w),
                "masks": [mask_a, mask_b],
                "positions": [],
            }
        )

    input_occ = np.zeros((h, w), dtype=bool)
    for spec in sorted(specs, key=lambda item: item["shape"][0] * item["shape"][1], reverse=True):
        rh, rw = spec["shape"]
        for mask in spec["masks"]:
            placed = False
            for _ in range(_PLACEMENT_ATTEMPTS):
                top = _randint(0, h - rh)
                left = _randint(0, w - rw)
                if not _can_place(input_occ, top, left, rh, rw, margin=1):
                    continue
                _mark_occ(input_occ, top, left, rh, rw, margin=1)
                spec["positions"].append((top, left, mask))
                placed = True
                break
            if not placed:
                return None

    ordered_specs = []
    for spec in specs:
        first_top, first_left, _ = min(spec["positions"], key=lambda item: (item[0], item[1]))
        ordered_specs.append((first_top, first_left, spec))
    ordered_specs.sort(key=lambda item: (item[0], item[1]))

    marker_centers: list[tuple[int, int]] = []
    cross_occ = input_occ.copy()
    output_occ = np.zeros((h, w), dtype=bool)
    prev_key = (-1, -1)
    for idx, (_, _, spec) in enumerate(ordered_specs):
        rh, rw = spec["shape"]
        candidates = []
        row_lo = rh // 2
        row_hi = h - rh // 2
        col_lo = rw // 2
        col_hi = w - rw // 2
        for cr in range(row_lo, row_hi):
            for cc in range(col_lo, col_hi):
                key = (cr, cc)
                if key <= prev_key:
                    continue
                top = cr - rh // 2
                left = cc - rw // 2
                if not _can_place(output_occ, top, left, rh, rw, margin=1):
                    continue
                if not _can_place(cross_occ, cr - 1, cc - 1, 3, 3, margin=1):
                    continue
                candidates.append((cr, cc))
        if not candidates:
            return None
        center = candidates[_randint(0, len(candidates) - 1)]
        marker_centers.append(center)
        prev_key = center
        top = center[0] - rh // 2
        left = center[1] - rw // 2
        _mark_occ(output_occ, top, left, rh, rw, margin=1)
        _mark_occ(cross_occ, center[0] - 1, center[1] - 1, 3, 3, margin=1)
        if idx == len(ordered_specs) - 1:
            break

    grid = np.full((h, w), bg, dtype=int)
    for spec in specs:
        for top, left, mask in spec["positions"]:
            _paint_mask(grid, top, left, mask, spec["color"])
    for center in marker_centers:
        _paint_cross(grid, center, marker_color)

    expected = np.full((h, w), bg, dtype=int)
    for (_, _, spec), center in zip(ordered_specs, marker_centers):
        rh, rw = spec["shape"]
        top = center[0] - rh // 2
        left = center[1] - rw // 2
        expected[top : top + rh, left : left + rw] = spec["color"]

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        result = _generate_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input")
