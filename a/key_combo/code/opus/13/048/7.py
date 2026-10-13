from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

SEGMENT_COLORS = (BLUE, RED, GREEN, YELLOW)
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]

SHORT_AXIS_RANGE = (12, 16)
LONG_AXIS_RANGE = (14, 18)
PIVOT_COUNT_CHOICES = (3, 4)
LENGTH_RANGE = (2, 4)
GEN_RETRIES = 800
INNER_RETRIES = 250


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < h and 0 <= int(c) < w


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotate_cw(dr: int, dc: int) -> tuple[int, int]:
    return int(dc), int(-dr)


def _line_coords(pivot: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = map(int, pivot)
    dr, dc = map(int, direction)
    return [(r + k * dr, c + k * dc) for k in range(1, int(length) + 1)]


def _line_patch(color: int, length: int, direction: tuple[int, int], bg: int) -> np.ndarray:
    dr, dc = map(int, direction)
    if dr == 0:
        patch = np.full((1, int(length)), int(bg), dtype=int)
        patch[0, :] = int(color)
    else:
        patch = np.full((int(length), 1), int(bg), dtype=int)
        patch[:, 0] = int(color)
    return patch


def _assemble_patches(patches: list[np.ndarray], axis: str, bg: int) -> np.ndarray:
    assert patches
    if axis == "row":
        out_h = sum(int(p.shape[0]) for p in patches)
        out_w = max(int(p.shape[1]) for p in patches)
        out = np.full((out_h, out_w), int(bg), dtype=int)
        cursor = 0
        for patch in patches:
            h, w = map(int, patch.shape)
            out[cursor : cursor + h, 0:w] = patch
            cursor += h
        return out

    out_h = max(int(p.shape[0]) for p in patches)
    out_w = sum(int(p.shape[1]) for p in patches)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    cursor = 0
    for patch in patches:
        h, w = map(int, patch.shape)
        out[0:h, cursor : cursor + w] = patch
        cursor += w
    return out


def _extract_segments(grid: np.ndarray) -> tuple[int, str, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)

    pivot_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == GREY)]
    assert len(pivot_coords) in PIVOT_COUNT_CHOICES
    pivot_set = set(pivot_coords)

    segments: list[dict] = []
    seen_colors: set[int] = set()
    expected_fg = set(pivot_coords)

    for pivot in pivot_coords:
        r, c = pivot
        adjacent: list[tuple[int, int, int]] = []
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not _in_bounds(g.shape, nr, nc):
                continue
            color = int(g[nr, nc])
            if color not in (bg, GREY):
                adjacent.append((dr, dc, color))
        assert len(adjacent) == 1
        dr, dc, color = adjacent[0]
        assert color in SEGMENT_COLORS
        assert color not in seen_colors
        seen_colors.add(color)

        src_coords: list[tuple[int, int]] = []
        nr, nc = r + dr, c + dc
        while _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == color:
            src_coords.append((nr, nc))
            nr += dr
            nc += dc
        assert LENGTH_RANGE[0] <= len(src_coords) <= LENGTH_RANGE[1]
        assert set(src_coords) == set(_line_coords(pivot, (dr, dc), len(src_coords)))

        color_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        assert set(color_coords) == set(src_coords)

        rot_dir = _rotate_cw(dr, dc)
        rot_coords = _line_coords(pivot, rot_dir, len(src_coords))
        for rr, cc in rot_coords:
            assert _in_bounds(g.shape, rr, cc)
            assert int(g[rr, cc]) == bg

        seg = {
            "pivot": pivot,
            "color": int(color),
            "dir": (int(dr), int(dc)),
            "length": int(len(src_coords)),
            "src": src_coords,
            "rot_dir": rot_dir,
            "rot": rot_coords,
            "adjacent": src_coords[0],
        }
        segments.append(seg)
        expected_fg.update(src_coords)

    actual_fg = {tuple(map(int, rc)) for rc in np.argwhere(g != bg)}
    assert actual_fg == expected_fg

    rows = {r for r, _c in pivot_coords}
    cols = {c for _r, c in pivot_coords}
    same_row = len(rows) == 1 and len(cols) > 1
    same_col = len(cols) == 1 and len(rows) > 1
    assert same_row != same_col
    axis = "row" if same_row else "col"
    return int(bg), axis, segments


def _transformed_grid(grid: np.ndarray, bg: int, segments: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for seg in segments:
        for r, c in seg["src"]:
            out[r, c] = GREY
        for r, c in seg["rot"]:
            assert int(out[r, c]) == int(bg)
            out[r, c] = int(seg["color"])
    return out


def _expected_from_segments(segments: list[dict], axis: str, bg: int) -> np.ndarray:
    if axis == "row":
        ordered = sorted(segments, key=lambda seg: (seg["pivot"][1], seg["pivot"][0]))
    else:
        ordered = sorted(segments, key=lambda seg: (seg["pivot"][0], seg["pivot"][1]))
    patches = [_line_patch(seg["color"], seg["length"], seg["rot_dir"], bg) for seg in ordered]
    return _assemble_patches(patches, axis, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each pivoted line clockwise, then crop and concatenate the rotated bars in pivot order.

    Input structure:
    - A black-dominant canvas contains 3 or 4 grey single-pixel pivots.
    - Each pivot touches exactly one straight colored bar of length 2 to 4, using distinct colors.
    - The pivots all lie on one common row or one common column, while the bars may point in different cardinal directions.

    Rule:
    - For each grey pivot, read the adjacent bar direction and rotate that bar 90 degrees clockwise around the pivot.
    - The original bar cells become grey traces, and the rotated bar is drawn in the bar's original color.
    - Sort pivots left-to-right when they share a row, or top-to-bottom when they share a column.
    - Crop each rotated bar to its own minimal bounding box, discard pivots and traces, and concatenate the cropped bars in that sorted order.
    - Row-aligned pivots produce a vertical stack of cropped bars; column-aligned pivots produce a horizontal strip.

    Key insight:
    - The hidden organizing signal is the pivots' collinearity: it determines both the sorting axis and the direction in which the rotated bar crops are assembled.
    """
    g = np.asarray(grid, dtype=int)
    bg, axis, segments = _extract_segments(g)
    transformed = _transformed_grid(g, bg, segments)

    if axis == "row":
        ordered = sorted(segments, key=lambda seg: (seg["pivot"][1], seg["pivot"][0]))
    else:
        ordered = sorted(segments, key=lambda seg: (seg["pivot"][0], seg["pivot"][1]))

    patches = []
    for seg in ordered:
        patch = _crop_bbox(transformed, _bbox_from_coords(seg["rot"]))
        patches.append(patch)
    return _assemble_patches(patches, axis, bg)


def _sample_positions(low: int, high: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(int(low), int(high) + 1))
    if len(values) < count:
        return None
    for _ in range(120):
        picked = sorted(random.sample(values, count))
        if all(b - a >= int(min_gap) for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _allowed_dirs(shape: tuple[int, int], pivot: tuple[int, int], length: int) -> list[tuple[int, int]]:
    allowed: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        src = _line_coords(pivot, (dr, dc), length)
        rot = _line_coords(pivot, _rotate_cw(dr, dc), length)
        if all(_in_bounds(shape, r, c) for r, c in src + rot):
            allowed.append((dr, dc))
    return allowed


def _input_layout_valid(specs: list[dict], shape: tuple[int, int]) -> bool:
    pivot_set = {spec["pivot"] for spec in specs}
    src_union: set[tuple[int, int]] = set()
    rot_union: set[tuple[int, int]] = set()

    for i, spec in enumerate(specs):
        own_adjacent = spec["src"][0]
        for cell in spec["src"]:
            if cell in pivot_set:
                return False
            if cell in src_union:
                return False
            src_union.add(cell)
            for other in specs:
                if other["pivot"] != spec["pivot"]:
                    if abs(cell[0] - other["pivot"][0]) + abs(cell[1] - other["pivot"][1]) == 1:
                        return False

        for cell in spec["rot"]:
            if cell in pivot_set or cell in src_union or cell in rot_union:
                return False
            if not _in_bounds(shape, cell[0], cell[1]):
                return False
            rot_union.add(cell)

        obj_cells = {spec["pivot"], *spec["src"]}
        for other in specs[i + 1 :]:
            other_cells = {other["pivot"], *other["src"]}
            for a in obj_cells:
                for b in other_cells:
                    if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                        return False

        assert own_adjacent not in pivot_set

    return True


def _reject_shortcuts(grid: np.ndarray) -> bool:
    bg, axis, segments = _extract_segments(grid)
    ordered = sorted(segments, key=lambda seg: (seg["pivot"][1], seg["pivot"][0])) if axis == "row" else sorted(
        segments, key=lambda seg: (seg["pivot"][0], seg["pivot"][1])
    )

    colors = [seg["color"] for seg in ordered]
    if colors == sorted(colors) or colors == sorted(colors, reverse=True):
        return False

    expected = _expected_from_segments(segments, axis, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if max(actual.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice(("row", "col"))
    count = random.choice(PIVOT_COUNT_CHOICES)
    if axis == "row":
        h = random.randint(*SHORT_AXIS_RANGE)
        w = random.randint(*LONG_AXIS_RANGE)
    else:
        h = random.randint(*LONG_AXIS_RANGE)
        w = random.randint(*SHORT_AXIS_RANGE)

    if axis == "row":
        shared = random.randint(2, h - 3)
        positions = _sample_positions(2, w - 3, count, 2)
        if positions is None:
            return None
        pivots = [(shared, c) for c in positions]
    else:
        shared = random.randint(2, w - 3)
        positions = _sample_positions(2, h - 3, count, 2)
        if positions is None:
            return None
        pivots = [(r, shared) for r in positions]

    colors = random.sample(list(SEGMENT_COLORS), count)

    for _ in range(INNER_RETRIES):
        lengths = [random.randint(*LENGTH_RANGE) for _ in range(count)]
        if len(set(lengths)) < 2:
            continue

        allowed = [_allowed_dirs((h, w), pivot, length) for pivot, length in zip(pivots, lengths)]
        if any(not choices for choices in allowed):
            continue

        dirs = [random.choice(choices) for choices in allowed]
        if not any(dr == 0 for dr, _dc in dirs):
            continue
        if not any(dc == 0 for _dr, dc in dirs):
            continue

        specs = []
        for pivot, color, length, direction in zip(pivots, colors, lengths, dirs):
            specs.append(
                {
                    "pivot": pivot,
                    "color": color,
                    "length": length,
                    "dir": direction,
                    "src": _line_coords(pivot, direction, length),
                    "rot_dir": _rotate_cw(*direction),
                    "rot": _line_coords(pivot, _rotate_cw(*direction), length),
                }
            )
        if not _input_layout_valid(specs, (h, w)):
            continue

        grid = np.full((h, w), BLACK, dtype=int)
        for spec in specs:
            pr, pc = spec["pivot"]
            grid[pr, pc] = GREY
            for r, c in spec["src"]:
                grid[r, c] = spec["color"]

        try:
            if not _reject_shortcuts(grid):
                continue
        except Exception:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
