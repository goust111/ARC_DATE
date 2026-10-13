from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
TEAL = 8

_DIR_TO_VEC = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(grid).ravel())
    return int(counts.most_common(1)[0][0])


def _components_of_color(grid: np.ndarray, color: int, *, connectivity: int = 4) -> list[np.ndarray]:
    g = np.asarray(grid)
    mask = g == int(color)
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
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
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _bar_info(coords: np.ndarray) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    rows = sorted(set(int(r) for r in coords[:, 0]))
    cols = sorted(set(int(c) for c in coords[:, 1]))
    if len(rows) == 1:
        length = c1 - c0 + 1
        expected = {(rows[0], c) for c in range(c0, c1 + 1)}
        assert len(coords) == length and expected == set(map(tuple, coords)), "blue object must be a solid horizontal bar"
        return {
            "orientation": "H",
            "row": rows[0],
            "c0": c0,
            "c1": c1,
            "length": length,
            "radius": (length - 1) // 2,
            "center": (c0 + c1) / 2.0,
            "endpoints": ((rows[0], c0), (rows[0], c1)),
        }
    if len(cols) == 1:
        length = r1 - r0 + 1
        expected = {(r, cols[0]) for r in range(r0, r1 + 1)}
        assert len(coords) == length and expected == set(map(tuple, coords)), "blue object must be a solid vertical bar"
        return {
            "orientation": "V",
            "col": cols[0],
            "r0": r0,
            "r1": r1,
            "length": length,
            "radius": (length - 1) // 2,
            "center": (r0 + r1) / 2.0,
            "endpoints": ((r0, cols[0]), (r1, cols[0])),
        }
    raise AssertionError("blue object must be one cell thick")


def _infer_direction(info: dict[str, object], markers: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    if info["orientation"] == "H":
        row = int(info["row"])
        left = (row, int(info["c0"]))
        right = (row, int(info["c1"]))
        left_marks = {(row - 1, left[1]), (row + 1, left[1])} & markers
        right_marks = {(row - 1, right[1]), (row + 1, right[1])} & markers
        assert (len(left_marks) == 1) ^ (len(right_marks) == 1), "each bar must have exactly one endpoint marker"
        if left_marks:
            return _DIR_TO_VEC["right"], next(iter(left_marks))
        return _DIR_TO_VEC["left"], next(iter(right_marks))

    col = int(info["col"])
    top = (int(info["r0"]), col)
    bottom = (int(info["r1"]), col)
    top_marks = {(top[0], col - 1), (top[0], col + 1)} & markers
    bottom_marks = {(bottom[0], col - 1), (bottom[0], col + 1)} & markers
    assert (len(top_marks) == 1) ^ (len(bottom_marks) == 1), "each bar must have exactly one endpoint marker"
    if top_marks:
        return _DIR_TO_VEC["down"], next(iter(top_marks))
    return _DIR_TO_VEC["up"], next(iter(bottom_marks))


def _analyze(grid: np.ndarray) -> tuple[int, list[dict[str, object]], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert not np.any(g == TEAL), "teal must be absent from the input"

    bars = _components_of_color(g, BLUE, connectivity=4)
    markers = {tuple(map(int, rc)) for rc in np.argwhere(g == GREY)}
    assert bars, "expected at least one blue bar"
    assert len(markers) == len(bars), "expected one grey marker per blue bar"

    used_markers: set[tuple[int, int]] = set()
    analyses: list[dict[str, object]] = []
    for bar_coords in bars:
        info = _bar_info(bar_coords)
        assert 5 <= int(info["length"]) <= 9, "bar length must be within the task range"
        direction, marker = _infer_direction(info, markers)
        assert marker not in used_markers, "marker must belong to exactly one bar"
        used_markers.add(marker)
        info["direction"] = direction
        info["marker"] = marker
        analyses.append(info)

    assert used_markers == markers, "all grey markers must be assigned"
    return bg, analyses, markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a one-sided teal diamond envelope around each marked blue bar.

    Input structure:
    - The grid is a black background containing 2-4 blue one-cell-thick bars.
    - Every bar is horizontal or vertical, solid, and has length 5-9.
    - Exactly one grey pixel sits beside one endpoint, perpendicular to the bar,
      marking that endpoint as the tail.

    Rule:
    - Infer each bar's direction from the grey tail marker toward the opposite
      unmarked endpoint.
    - Build the centered diamond envelope that the full bar would generate: each
      step away from the bar shortens the parallel segment by one cell at both ends.
    - Keep only the half of that envelope lying on the head side of the bar's
      geometric center, including the center line, and paint those background
      cells teal while leaving the original blue bars and grey markers unchanged.

    Key insight:
    - The grey pixel is not part of the envelope; it breaks the bar's symmetry by
      identifying the tail, which decides which half of an otherwise symmetric
      diamond survives.
    """
    g = np.asarray(grid, dtype=int)
    bg, analyses, _markers = _analyze(g)
    out = g.copy()
    H, W = g.shape

    for info in analyses:
        length = int(info["length"])
        radius = int(info["radius"])
        dr, dc = info["direction"]  # type: ignore[misc]

        if info["orientation"] == "H":
            row = int(info["row"])
            c0 = int(info["c0"])
            c1 = int(info["c1"])
            center = float(info["center"])
            for offset in range(1, radius + 1):
                left = c0 + offset
                right = c1 - offset
                for rr in (row - offset, row + offset):
                    assert 0 <= rr < H
                    for cc in range(left, right + 1):
                        if (cc - center) * dc >= 0 and out[rr, cc] == bg:
                            out[rr, cc] = TEAL
        else:
            col = int(info["col"])
            r0 = int(info["r0"])
            r1 = int(info["r1"])
            center = float(info["center"])
            for offset in range(1, radius + 1):
                top = r0 + offset
                bottom = r1 - offset
                for cc in (col - offset, col + offset):
                    assert 0 <= cc < W
                    for rr in range(top, bottom + 1):
                        if (rr - center) * dr >= 0 and out[rr, cc] == bg:
                            out[rr, cc] = TEAL

    return out


_RNG = np.random.default_rng()
_GRID_MIN = 16
_GRID_MAX = 25
_COUNT_CHOICES = np.array([2, 3, 4], dtype=int)
_COUNT_WEIGHTS = np.array([1, 2, 3], dtype=float)
_LENGTHS = np.array([5, 6, 7, 8, 9], dtype=int)
_MAX_ATTEMPTS = 400


def _sample_count() -> int:
    probs = _COUNT_WEIGHTS / _COUNT_WEIGHTS.sum()
    return int(_RNG.choice(_COUNT_CHOICES, p=probs))


def _sample_specs() -> list[dict[str, object]]:
    count = _sample_count()
    directions = [
        str(_RNG.choice(["left", "right"])),
        str(_RNG.choice(["up", "down"])),
    ]
    while len(directions) < count:
        directions.append(str(_RNG.choice(["left", "right", "up", "down"])))
    _RNG.shuffle(directions)

    lengths = [int(_RNG.choice(_LENGTHS)) for _ in range(count)]
    if not any(length % 2 == 0 for length in lengths):
        lengths[0] = int(_RNG.choice([6, 8]))
    if not any(length % 2 == 1 for length in lengths):
        lengths[-1] = int(_RNG.choice([5, 7, 9]))

    specs = []
    for direction, length in zip(directions, lengths):
        radius = (length - 1) // 2
        if direction in {"left", "right"}:
            bbox_h = 2 * radius + 1
            bbox_w = length
        else:
            bbox_h = length
            bbox_w = 2 * radius + 1
        specs.append(
            {
                "direction": direction,
                "length": length,
                "radius": radius,
                "bbox_area": bbox_h * bbox_w,
            }
        )
    return sorted(specs, key=lambda item: (-int(item["bbox_area"]), -int(item["length"])))


def _candidate_iter(height: int, width: int, spec: dict[str, object]) -> list[tuple[int, int, int]]:
    length = int(spec["length"])
    radius = int(spec["radius"])
    direction = str(spec["direction"])
    candidates: list[tuple[int, int, int]] = []

    if direction in {"left", "right"}:
        for row in range(radius, height - radius):
            for c0 in range(0, width - length + 1):
                for side in (-1, 1):
                    marker_r = row + side
                    if 0 <= marker_r < height:
                        candidates.append((row, c0, side))
    else:
        for r0 in range(0, height - length + 1):
            for col in range(radius, width - radius):
                for side in (-1, 1):
                    marker_c = col + side
                    if 0 <= marker_c < width:
                        candidates.append((r0, col, side))

    order = _RNG.permutation(len(candidates))
    return [candidates[int(i)] for i in order]


def _bar_cells(spec: dict[str, object], anchor: tuple[int, int, int]) -> list[tuple[int, int]]:
    direction = str(spec["direction"])
    length = int(spec["length"])
    a0, a1, _side = anchor
    if direction in {"left", "right"}:
        return [(a0, c) for c in range(a1, a1 + length)]
    return [(r, a1) for r in range(a0, a0 + length)]


def _marker_cell(spec: dict[str, object], anchor: tuple[int, int, int]) -> tuple[int, int]:
    direction = str(spec["direction"])
    length = int(spec["length"])
    a0, a1, side = anchor
    if direction == "right":
        return a0 + side, a1
    if direction == "left":
        return a0 + side, a1 + length - 1
    if direction == "down":
        return a0, a1 + side
    return a0 + length - 1, a1 + side


def _diamond_bbox(spec: dict[str, object], anchor: tuple[int, int, int]) -> tuple[int, int, int, int]:
    direction = str(spec["direction"])
    length = int(spec["length"])
    radius = int(spec["radius"])
    a0, a1, _side = anchor
    if direction in {"left", "right"}:
        return a0 - radius, a1, a0 + radius, a1 + length - 1
    return a0, a1 - radius, a0 + length - 1, a1 + radius


def _neighborhood8(cell: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = shape
    r, c = cell
    cells = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < H and 0 <= nc < W:
                cells.append((nr, nc))
    return cells


def _can_place(
    shape: tuple[int, int],
    spec: dict[str, object],
    anchor: tuple[int, int, int],
    diamond_blocked: np.ndarray,
    input_occupied: np.ndarray,
    bar_keepout: np.ndarray,
    marker_keepout: np.ndarray,
) -> bool:
    r0, c0, r1, c1 = _diamond_bbox(spec, anchor)
    if diamond_blocked[r0 : r1 + 1, c0 : c1 + 1].any():
        return False

    marker = _marker_cell(spec, anchor)
    if diamond_blocked[marker]:
        return False

    cells = _bar_cells(spec, anchor) + [marker]
    for cell in cells:
        if input_occupied[cell] or marker_keepout[cell]:
            return False
    for cell in _bar_cells(spec, anchor):
        if bar_keepout[cell]:
            return False

    for nr, nc in _neighborhood8(marker, shape):
        if input_occupied[nr, nc]:
            return False
    return True


def _apply_placement(
    grid: np.ndarray,
    spec: dict[str, object],
    anchor: tuple[int, int, int],
    diamond_blocked: np.ndarray,
    input_occupied: np.ndarray,
    bar_keepout: np.ndarray,
    marker_keepout: np.ndarray,
) -> None:
    r0, c0, r1, c1 = _diamond_bbox(spec, anchor)
    diamond_blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    for cell in _bar_cells(spec, anchor):
        grid[cell] = BLUE
        input_occupied[cell] = True
        r, c = cell
        for nr, nc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                bar_keepout[nr, nc] = True

    marker = _marker_cell(spec, anchor)
    grid[marker] = GREY
    input_occupied[marker] = True
    for nr, nc in _neighborhood8(marker, grid.shape):
        marker_keepout[nr, nc] = True


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        bg, analyses, markers = _analyze(grid)
    except AssertionError:
        return False
    if bg != BLACK:
        return False
    if not (2 <= len(analyses) <= 4):
        return False

    orientations = {str(info["orientation"]) for info in analyses}
    if orientations != {"H", "V"}:
        return False

    bboxes = []
    for info in analyses:
        if info["orientation"] == "H":
            row = int(info["row"])
            c0 = int(info["c0"])
            c1 = int(info["c1"])
            radius = int(info["radius"])
            bbox = (row - radius, c0, row + radius, c1)
            own_bar_cells = {(row, c) for c in range(c0, c1 + 1)}
        else:
            col = int(info["col"])
            r0 = int(info["r0"])
            r1 = int(info["r1"])
            radius = int(info["radius"])
            bbox = (r0, col - radius, r1, col + radius)
            own_bar_cells = {(r, col) for r in range(r0, r1 + 1)}
        bboxes.append(bbox)

        marker = info["marker"]  # type: ignore[assignment]
        for nr, nc in _neighborhood8(marker, grid.shape):
            if (nr, nc) == marker:
                continue
            if (nr, nc) in own_bar_cells:
                continue
            if grid[nr, nc] in {BLUE, GREY}:
                return False

    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _rectangles_overlap(bboxes[i], bboxes[j]):
                return False

    for marker in markers:
        if grid[marker] != GREY:
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
    width = int(_RNG.integers(_GRID_MIN, _GRID_MAX + 1))
    grid = np.full((height, width), BLACK, dtype=int)

    diamond_blocked = np.zeros((height, width), dtype=bool)
    input_occupied = np.zeros((height, width), dtype=bool)
    bar_keepout = np.zeros((height, width), dtype=bool)
    marker_keepout = np.zeros((height, width), dtype=bool)

    for spec in _sample_specs():
        placed = False
        for anchor in _candidate_iter(height, width, spec):
            if _can_place((height, width), spec, anchor, diamond_blocked, input_occupied, bar_keepout, marker_keepout):
                _apply_placement(grid, spec, anchor, diamond_blocked, input_occupied, bar_keepout, marker_keepout)
                placed = True
                break
        if not placed:
            return None

    if not _validate_generated_input(grid):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.any((grid != BLACK) & (out == TEAL)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid direction-truncated-envelope input")
