from __future__ import annotations

import numpy as np


BG = 0
GRAY = 5
CONTROL_COLORS = np.array([1, 2, 3, 6, 7, 8, 9], dtype=int)
MARKER_LENGTHS = np.array([1, 2, 3], dtype=int)
GEN_ATTEMPTS = 500


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _center_distance2(shape: tuple[int, int], r: int, c: int) -> int:
    """Twice the Chebyshev distance to the geometric grid center."""
    return max(abs(2 * int(r) - (int(shape[0]) - 1)), abs(2 * int(c) - (int(shape[1]) - 1)))


def _ring_color(control_color: int, distance: int) -> int:
    return int(control_color) if distance % 2 == 0 else GRAY


def _parse_controls(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    controls: list[dict[str, int]] = []

    for color in colors:
        coords = np.argwhere(g == color)
        assert 2 <= len(coords) <= 4, "each control color must form one control pixel plus a short right marker"
        rows = np.unique(coords[:, 0])
        assert len(rows) == 1, "each control color must lie on a single row"
        row = int(rows[0])
        cols = sorted(int(c) for c in coords[:, 1])
        start = cols[0]
        assert cols == list(range(start, start + len(cols))), "marker must be a contiguous horizontal line"

        marker_len = len(cols) - 1
        assert 1 <= marker_len <= 3, "marker length must be between 1 and 3"
        controls.append(
            {
                "row": row,
                "col": start,
                "color": color,
                "marker_len": marker_len,
                "radius": marker_len + 1,
                "priority": _center_distance2(g.shape, row, start),
            }
        )

    priorities = [control["priority"] for control in controls]
    assert len(priorities) in (2, 3), "expected 2 or 3 controls"
    assert len(set(priorities)) == len(priorities), "control priorities must be unique"
    return controls


def _render_from_controls(grid: np.ndarray, controls: list[dict[str, int]]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for r in range(h):
        for c in range(w):
            best_priority = None
            best_color = None
            for control in controls:
                dist = max(abs(r - control["row"]), abs(c - control["col"]))
                if dist > control["radius"]:
                    continue
                priority = control["priority"]
                if best_priority is None or priority < best_priority:
                    best_priority = priority
                    best_color = _ring_color(control["color"], dist)
            if best_color is not None:
                out[r, c] = best_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay finite concentric square rings from several control pixels, letting the control closer to the grid center win overlaps.

    Input structure:
    - The grid is mostly black background.
    - There are 2 or 3 distinct non-black control colors.
    - Each control consists of a leftmost control pixel plus a same-colored horizontal marker line immediately to its right.

    Rule:
    - For each control, read the marker-line length L and set its maximum ring radius to L+1 in Chebyshev distance.
    - Around each control pixel, draw square rings from distance 0 through L+1, using the control color at distance 0 and every even distance, and gray at every odd distance.
    - When multiple controls cover the same cell, keep the ring color from the control whose pixel has the smallest Chebyshev distance to the grid center.
    - Cells not covered by any ring stay unchanged.

    Key insight:
    - The right-side marker encodes how far each concentric pattern expands, while overlap resolution depends on a global spatial priority: whichever control sits closer to the overall grid center dominates local conflicts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    controls = _parse_controls(g)
    return _render_from_controls(g, controls)


def _occupied_run(center: tuple[int, int], marker_len: int) -> set[tuple[int, int]]:
    r, c = center
    return {(int(r), int(c + offset)) for offset in range(marker_len + 1)}


def _candidate_centers(height: int, width: int, radius: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(radius, height - radius)
        for c in range(radius, width - radius)
    ]


def _conflict_stats(shape: tuple[int, int], controls: list[dict[str, int]]) -> tuple[int, int, int]:
    overlap_cells = 0
    conflict_cells = 0
    covered_cells = 0
    for r in range(shape[0]):
        for c in range(shape[1]):
            covers = []
            for control in controls:
                dist = max(abs(r - control["row"]), abs(c - control["col"]))
                if dist <= control["radius"]:
                    covers.append((control["priority"], _ring_color(control["color"], dist)))
            if covers:
                covered_cells += 1
            if len(covers) >= 2:
                overlap_cells += 1
                covers.sort()
                if any(color != covers[0][1] for _, color in covers[1:]):
                    conflict_cells += 1
    return overlap_cells, conflict_cells, covered_cells


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        controls = _parse_controls(grid)
    except AssertionError:
        return False

    shape = tuple(int(v) for v in grid.shape)
    centers = [(control["row"], control["col"]) for control in controls]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if _chebyshev(centers[i], centers[j]) < 4:
                return False

    has_overlap_pair = False
    for i in range(len(controls)):
        for j in range(i + 1, len(controls)):
            if _chebyshev(centers[i], centers[j]) <= controls[i]["radius"] + controls[j]["radius"]:
                has_overlap_pair = True
    if not has_overlap_pair:
        return False

    overlap_cells, conflict_cells, covered_cells = _conflict_stats(shape, controls)
    area = shape[0] * shape[1]
    if overlap_cells == 0 or conflict_cells == 0:
        return False
    if covered_cells <= area // 8 or covered_cells >= (area * 11) // 20:
        return False
    if overlap_cells >= area // 4:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.count_nonzero(out == GRAY) == 0:
        return False
    return True


def _pick_from(candidates: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not candidates:
        return None
    idx = int(np.random.randint(0, len(candidates)))
    return candidates[idx]


def _can_use_center(
    center: tuple[int, int],
    marker_len: int,
    existing_controls: list[dict[str, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    run_cells = _occupied_run(center, marker_len)
    if any(cell in occupied for cell in run_cells):
        return False

    priority = _center_distance2(shape, center[0], center[1])
    if any(priority == control["priority"] for control in existing_controls):
        return False

    for control in existing_controls:
        if _chebyshev(center, (control["row"], control["col"])) < 4:
            return False
    return True


def _build_input_from_controls(
    shape: tuple[int, int], controls: list[dict[str, int]]
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for control in controls:
        r = control["row"]
        c = control["col"]
        color = control["color"]
        marker_len = control["marker_len"]
        grid[r, c : c + marker_len + 1] = color
    return grid


def _generate_attempt() -> np.ndarray | None:
    height = int(np.random.randint(13, 20))
    width = int(np.random.randint(13, 20))
    n_controls = 2 if np.random.random() < 0.5 else 3
    marker_lens = [int(np.random.choice(MARKER_LENGTHS)) for _ in range(n_controls)]
    colors = [int(v) for v in np.random.choice(CONTROL_COLORS, size=n_controls, replace=False)]

    controls: list[dict[str, int]] = []
    occupied: set[tuple[int, int]] = set()

    first_marker = marker_lens[0]
    first_radius = first_marker + 1
    first_center = _pick_from(_candidate_centers(height, width, first_radius))
    if first_center is None:
        return None
    controls.append(
        {
            "row": first_center[0],
            "col": first_center[1],
            "color": colors[0],
            "marker_len": first_marker,
            "radius": first_radius,
            "priority": _center_distance2((height, width), *first_center),
        }
    )
    occupied |= _occupied_run(first_center, first_marker)

    second_marker = marker_lens[1]
    second_radius = second_marker + 1
    second_candidates = []
    for center in _candidate_centers(height, width, second_radius):
        if not _can_use_center(center, second_marker, controls, occupied, (height, width)):
            continue
        dist = _chebyshev(center, first_center)
        if dist <= first_radius + second_radius:
            second_candidates.append(center)
    second_center = _pick_from(second_candidates)
    if second_center is None:
        return None
    controls.append(
        {
            "row": second_center[0],
            "col": second_center[1],
            "color": colors[1],
            "marker_len": second_marker,
            "radius": second_radius,
            "priority": _center_distance2((height, width), *second_center),
        }
    )
    occupied |= _occupied_run(second_center, second_marker)

    if n_controls == 3:
        third_marker = marker_lens[2]
        third_radius = third_marker + 1
        require_overlap = np.random.random() < 0.6
        third_candidates = []
        for center in _candidate_centers(height, width, third_radius):
            if not _can_use_center(center, third_marker, controls, occupied, (height, width)):
                continue
            overlaps_any = any(
                _chebyshev(center, (control["row"], control["col"])) <= control["radius"] + third_radius
                for control in controls
            )
            if require_overlap and not overlaps_any:
                continue
            third_candidates.append(center)
        third_center = _pick_from(third_candidates)
        if third_center is None:
            return None
        controls.append(
            {
                "row": third_center[0],
                "col": third_center[1],
                "color": colors[2],
                "marker_len": third_marker,
                "radius": third_radius,
                "priority": _center_distance2((height, width), *third_center),
            }
        )

    grid = _build_input_from_controls((height, width), controls)
    if not _validate_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
