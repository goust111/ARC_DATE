from __future__ import annotations

import random

import numpy as np


BG = 0
BASE = 1
MARKER_POOL = (2, 3, 4)
MAX_GEN_ATTEMPTS = 500
MIN_REPEAT = 2
MAX_REPEAT = 3


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _normalized(points: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]]:
    rows = [r for r, _ in points.values()]
    cols = [c for _, c in points.values()]
    r0 = min(rows)
    c0 = min(cols)
    return {color: (r - r0, c - c0) for color, (r, c) in points.items()}


def _layout_signature(points: dict[int, tuple[int, int]]) -> tuple[tuple[int, int, int], ...]:
    norm = _normalized(points)
    return tuple(sorted((int(color), int(r), int(c)) for color, (r, c) in norm.items()))


def _run_starts(grid: np.ndarray, axis: int) -> list[int]:
    length = grid.shape[axis]
    starts = [0]
    for i in range(1, length):
        prev = grid[i - 1, :] if axis == 0 else grid[:, i - 1]
        cur = grid[i, :] if axis == 0 else grid[:, i]
        if not np.array_equal(cur, prev):
            starts.append(i)
    return starts


def _run_lengths(grid: np.ndarray, axis: int) -> list[int]:
    length = grid.shape[axis]
    lengths: list[int] = []
    cur = 1
    for i in range(1, length):
        prev = grid[i - 1, :] if axis == 0 else grid[:, i - 1]
        now = grid[i, :] if axis == 0 else grid[:, i]
        if np.array_equal(prev, now):
            cur += 1
        else:
            lengths.append(cur)
            cur = 1
    lengths.append(cur)
    return lengths


def _compress(grid: np.ndarray) -> np.ndarray:
    row_idx = _run_starts(grid, axis=0)
    col_idx = _run_starts(grid, axis=1)
    return grid[np.ix_(row_idx, col_idx)].copy()


def _marker_points(grid: np.ndarray, colors: list[int]) -> dict[int, tuple[int, int]]:
    points: dict[int, tuple[int, int]] = {}
    for color in colors:
        locs = np.argwhere(grid == color)
        if len(locs) != 1:
            raise AssertionError(f"expected exactly one cell for marker color {color}")
        points[int(color)] = (int(locs[0, 0]), int(locs[0, 1]))
    return points


def _matching_rotations(
    crop: np.ndarray, target_layout: dict[int, tuple[int, int]], colors: list[int]
) -> list[int]:
    matches: list[int] = []
    for k in range(4):
        rotated = np.rot90(crop, k)
        points = _marker_points(rotated, colors)
        if _normalized(points) == target_layout:
            matches.append(k)
    return matches


def _repeat_groups(grid: np.ndarray, axis: int) -> int:
    return sum(1 for length in _run_lengths(grid, axis) if length > 1)


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the blue rectangle, orient it from external markers, then compress duplicate row and column runs.

    Input structure:
    - A black background contains one solid blue rectangle whose interior also includes 2 or 3 singleton marker colors.
    - The same marker colors each appear once more as isolated external pixels in the background.
    - The rectangle has been row/column-expanded, so several consecutive rows and columns are exact duplicates.

    Rule:
    - Find the blue rectangle, crop it, and collect the singleton marker colors inside that crop together with their matching external pixels.
    - Normalize the inside-marker coordinates and the outside-marker coordinates by each marker set's top-left corner.
    - Choose the unique rotation among 0, 90, 180, and 270 degrees whose rotated inside-marker layout matches the external-marker layout.
    - Rotate the crop by that angle, then collapse consecutive identical rows to one row and consecutive identical columns to one column.

    Key insight:
    - The outside pixels are an orientation legend for the expanded rectangle; only after restoring that orientation does run-compression reveal the intended core pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rect_box = _bbox(g == BASE)
    assert rect_box is not None, "expected one blue base rectangle"
    r0, c0, r1, c1 = rect_box
    crop = _crop(g, rect_box)

    colors = sorted(int(color) for color in np.unique(g) if int(color) not in (BG, BASE))
    inside: dict[int, tuple[int, int]] = {}
    outside: dict[int, tuple[int, int]] = {}
    for color in colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 2, f"marker color {color} must appear exactly twice"
        inside_coords = []
        outside_coords = []
        for rr, cc in coords:
            rr = int(rr)
            cc = int(cc)
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                inside_coords.append((rr - r0, cc - c0))
            else:
                outside_coords.append((rr, cc))
        assert len(inside_coords) == 1 and len(outside_coords) == 1
        inside[color] = inside_coords[0]
        outside[color] = outside_coords[0]

    assert 2 <= len(inside) <= 3
    target_layout = _normalized(outside)
    marker_colors = sorted(inside)

    matches = _matching_rotations(crop, target_layout, marker_colors)
    assert len(matches) == 1, "expected a unique orientation match"
    oriented = np.rot90(crop, matches[0])
    return _compress(oriented)


def _make_core() -> tuple[np.ndarray, list[int]] | None:
    marker_count = random.choice((2, 3))
    slot_dims_by_count = {
        2: ((2, 2),),
        3: ((2, 2), (2, 3), (3, 2), (3, 3)),
    }
    slot_rows, slot_cols = random.choice(slot_dims_by_count[marker_count])
    h, w = 2 * slot_rows + 1, 2 * slot_cols + 1
    colors = sorted(random.sample(MARKER_POOL, marker_count))
    slots = [(sr, sc) for sr in range(slot_rows) for sc in range(slot_cols)]

    for _ in range(200):
        chosen = random.sample(slots, marker_count)
        used_rows = {sr for sr, _ in chosen}
        used_cols = {sc for _, sc in chosen}
        if len(used_rows) != slot_rows or len(used_cols) != slot_cols:
            continue

        core = np.full((h, w), BASE, dtype=int)
        for color, (sr, sc) in zip(colors, chosen):
            core[2 * sr + 1, 2 * sc + 1] = color

        if _bbox(core == BASE) != (0, 0, h - 1, w - 1):
            continue
        if not np.array_equal(_compress(core), core):
            continue

        layouts = []
        for k in range(4):
            rotated = np.rot90(core, k)
            points = _marker_points(rotated, colors)
            layouts.append(_layout_signature(points))
        if len(set(layouts)) != 4:
            continue
        return core, colors

    return None


def _make_axis_repeats(length: int) -> list[int]:
    while True:
        repeats = [
            random.randint(MIN_REPEAT, MAX_REPEAT) if idx % 2 == 0 else 1
            for idx in range(length)
        ]
        separator_repeats = repeats[::2]
        if len(set(separator_repeats)) > 1:
            return repeats


def _expand(core: np.ndarray, row_repeats: list[int], col_repeats: list[int]) -> np.ndarray:
    return np.repeat(np.repeat(core, row_repeats, axis=0), col_repeats, axis=1)


def _place_markers(
    grid: np.ndarray,
    origin: tuple[int, int],
    layout: dict[int, tuple[int, int]],
) -> None:
    r0, c0 = origin
    for color, (dr, dc) in layout.items():
        grid[r0 + dr, c0 + dc] = color


def _choose_canvas(
    rect_shape: tuple[int, int], ext_shape: tuple[int, int]
) -> tuple[int, int, tuple[int, int], tuple[int, int]] | None:
    rect_h, rect_w = rect_shape
    ext_h, ext_w = ext_shape
    margin = 1
    gap = 2
    arrangement = random.choice(("above", "below", "left", "right"))

    if arrangement in ("above", "below"):
        min_h = rect_h + ext_h + gap + 2 * margin
        min_w = max(rect_w, ext_w) + 2 * margin
    else:
        min_h = max(rect_h, ext_h) + 2 * margin
        min_w = rect_w + ext_w + gap + 2 * margin

    if min_h > 30 or min_w > 30:
        return None

    H = random.randint(min_h, min(30, min_h + 4))
    W = random.randint(min_w, min(30, min_w + 4))

    if arrangement == "above":
        rect_r = random.randint(ext_h + gap + margin, H - rect_h - margin)
        ext_r = random.randint(margin, rect_r - gap - ext_h)
        rect_c = random.randint(margin, W - rect_w - margin)
        ext_c = random.randint(margin, W - ext_w - margin)
    elif arrangement == "below":
        rect_r = random.randint(margin, H - rect_h - ext_h - gap - margin)
        ext_r = random.randint(rect_r + rect_h + gap, H - ext_h - margin)
        rect_c = random.randint(margin, W - rect_w - margin)
        ext_c = random.randint(margin, W - ext_w - margin)
    elif arrangement == "left":
        rect_c = random.randint(ext_w + gap + margin, W - rect_w - margin)
        ext_c = random.randint(margin, rect_c - gap - ext_w)
        rect_r = random.randint(margin, H - rect_h - margin)
        ext_r = random.randint(margin, H - ext_h - margin)
    else:
        rect_c = random.randint(margin, W - rect_w - ext_w - gap - margin)
        ext_c = random.randint(rect_c + rect_w + gap, W - ext_w - margin)
        rect_r = random.randint(margin, H - rect_h - margin)
        ext_r = random.randint(margin, H - ext_h - margin)

    rect_box = (rect_r, rect_c, rect_r + rect_h - 1, rect_c + rect_w - 1)
    ext_box = (ext_r, ext_c, ext_r + ext_h - 1, ext_c + ext_w - 1)
    if not _boxes_separated(rect_box, ext_box, gap=gap):
        return None
    return H, W, (rect_r, rect_c), (ext_r, ext_c)


def _validate_instance(
    full_grid: np.ndarray,
    expected_core: np.ndarray,
    marker_colors: list[int],
    target_layout: dict[int, tuple[int, int]],
    raw_crop: np.ndarray,
) -> bool:
    try:
        if not np.array_equal(main(full_grid), expected_core):
            return False

        matches = _matching_rotations(raw_crop, target_layout, marker_colors)
        if len(matches) != 1:
            return False
        oriented = np.rot90(raw_crop, matches[0])

        if _repeat_groups(oriented, axis=0) < 3:
            return False
        if _repeat_groups(oriented, axis=1) < 3:
            return False

        for k in range(4):
            compressed = _compress(np.rot90(raw_crop, k))
            if k == matches[0]:
                if not np.array_equal(compressed, expected_core):
                    return False
            elif np.array_equal(compressed, expected_core):
                return False

        compressed_first = _compress(raw_crop)
        if _matching_rotations(compressed_first, target_layout, marker_colors):
            return False
    except Exception:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    built = _make_core()
    if built is None:
        return None
    core, marker_colors = built
    row_repeats = _make_axis_repeats(core.shape[0])
    col_repeats = _make_axis_repeats(core.shape[1])
    expanded = _expand(core, row_repeats, col_repeats)

    if _repeat_groups(expanded, axis=0) < 3 or _repeat_groups(expanded, axis=1) < 3:
        return None

    target_layout = _normalized(_marker_points(expanded, marker_colors))
    ext_h = max(r for r, _ in target_layout.values()) + 1
    ext_w = max(c for _, c in target_layout.values()) + 1

    input_rotation = random.randrange(4)
    raw_crop = np.rot90(expanded, input_rotation)
    canvas = _choose_canvas(raw_crop.shape, (ext_h, ext_w))
    if canvas is None:
        return None
    H, W, rect_origin, ext_origin = canvas

    grid = np.zeros((H, W), dtype=int)
    rr, rc = rect_origin
    grid[rr : rr + raw_crop.shape[0], rc : rc + raw_crop.shape[1]] = raw_crop
    _place_markers(grid, ext_origin, target_layout)

    if not _validate_instance(grid, core, marker_colors, target_layout, raw_crop):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
