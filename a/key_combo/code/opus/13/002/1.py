import random

import numpy as np


BG = 0
FRAME = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_NAMES = ("top", "bottom", "left", "right")
HORIZONTAL_SIDES = ("left", "right")
VERTICAL_SIDES = ("top", "bottom")
MAX_GENERATION_ATTEMPTS = 400


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bands_from_lines(lines: list[int]) -> list[tuple[int, int]]:
    bands = []
    for a, b in zip(lines, lines[1:]):
        assert b - a >= 2, "slot interiors must be at least one cell thick"
        bands.append((a + 1, b - 1))
    return bands


def _find_band_index(value: int, bands: list[tuple[int, int]]) -> int:
    matches = [i for i, (lo, hi) in enumerate(bands) if lo <= value <= hi]
    assert len(matches) == 1, "marker must align with exactly one slot lane"
    return matches[0]


def _parse_frame(grid: np.ndarray) -> dict[str, object]:
    frame_mask = grid == FRAME
    assert np.any(frame_mask), "expected a grey lattice frame"
    r0, c0, r1, c1 = _bbox_from_mask(frame_mask)

    row_lines = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == FRAME)]
    col_lines = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == FRAME)]
    assert len(row_lines) == len(col_lines), "expected a square logical lattice"
    assert len(row_lines) in (4, 5), "expected a 3x3 or 4x4 lattice"
    assert row_lines[0] == r0 and row_lines[-1] == r1
    assert col_lines[0] == c0 and col_lines[-1] == c1

    rebuilt = np.zeros_like(frame_mask)
    for r in row_lines:
        rebuilt[r, c0 : c1 + 1] = True
    for c in col_lines:
        rebuilt[r0 : r1 + 1, c] = True
    assert np.array_equal(frame_mask[r0 : r1 + 1, c0 : c1 + 1], rebuilt[r0 : r1 + 1, c0 : c1 + 1])

    row_bands = _bands_from_lines(row_lines)
    col_bands = _bands_from_lines(col_lines)
    return {
        "bbox": (r0, c0, r1, c1),
        "row_lines": row_lines,
        "col_lines": col_lines,
        "row_bands": row_bands,
        "col_bands": col_bands,
    }


def _collect_markers(grid: np.ndarray, frame: dict[str, object]) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    h, w = grid.shape
    r0, c0, r1, c1 = frame["bbox"]
    row_bands = frame["row_bands"]
    col_bands = frame["col_bands"]

    marker_mask = (grid != BG) & (grid != FRAME)
    marker_coords = np.argwhere(marker_mask)
    assert len(marker_coords) > 0, "expected at least one edge marker"

    row_markers: dict[int, tuple[int, int]] = {}
    col_markers: dict[int, tuple[int, int]] = {}
    for r, c in marker_coords:
        r = int(r)
        c = int(c)
        edge_hits = int(r == 0) + int(r == h - 1) + int(c == 0) + int(c == w - 1)
        assert edge_hits == 1, "markers must be isolated edge pixels"

        color = int(grid[r, c])
        if c == 0:
            lane = _find_band_index(r, row_bands)
            assert lane not in row_markers, "at most one horizontal marker per row lane"
            row_markers[lane] = (color, c0 - 1)
        elif c == w - 1:
            lane = _find_band_index(r, row_bands)
            assert lane not in row_markers, "at most one horizontal marker per row lane"
            row_markers[lane] = (color, w - 2 - c1)
        elif r == 0:
            lane = _find_band_index(c, col_bands)
            assert lane not in col_markers, "at most one vertical marker per column lane"
            col_markers[lane] = (color, r0 - 1)
        else:
            assert r == h - 1
            lane = _find_band_index(c, col_bands)
            assert lane not in col_markers, "at most one vertical marker per column lane"
            col_markers[lane] = (color, h - 2 - r1)

    return row_markers, col_markers


def _fill_slots(
    grid: np.ndarray,
    frame: dict[str, object],
    row_markers: dict[int, tuple[int, int]],
    col_markers: dict[int, tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    row_bands = frame["row_bands"]
    col_bands = frame["col_bands"]

    for sr, (r_lo, r_hi) in enumerate(row_bands):
        for sc, (c_lo, c_hi) in enumerate(col_bands):
            horizontal = row_markers.get(sr)
            vertical = col_markers.get(sc)
            if horizontal is None and vertical is None:
                continue
            if horizontal is not None and vertical is None:
                color = horizontal[0]
            elif vertical is not None and horizontal is None:
                color = vertical[0]
            else:
                assert horizontal is not None and vertical is not None
                if horizontal[1] <= vertical[1]:
                    color = horizontal[0]
                else:
                    color = vertical[0]
            out[r_lo : r_hi + 1, c_lo : c_hi + 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill lattice slots with edge-ray colors, using marker-to-frame distance to resolve row/column crossings.

    Input structure:
    - The grid has a black background, one grey one-pixel-thick rectangular lattice frame, and isolated single-pixel colored markers on the outer border.
    - The lattice divides its interior into a 3x3 or 4x4 array of black rectangular slots.
    - Each left/right marker aligns with exactly one slot row, and each top/bottom marker aligns with exactly one slot column.

    Rule:
    - A left/right marker colors every slot in its aligned row lane, and a top/bottom marker colors every slot in its aligned column lane.
    - If a slot is touched by both a horizontal and a vertical ray, compare each marker's background gap to the nearest frame pixel along the ray direction.
    - The smaller distance wins; if the distances are equal, the horizontal ray wins; untouched slots and the grey frame stay unchanged.

    Key insight:
    - The border marker positions encode both which slot lane is activated and the priority used when horizontal and vertical rays compete inside the lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    frame = _parse_frame(g)
    row_markers, col_markers = _collect_markers(g, frame)
    return _fill_slots(g, frame, row_markers, col_markers)


def _other(side: str) -> str:
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    if side == "top":
        return "bottom"
    assert side == "bottom"
    return "top"


def _sample_slot_sizes(n_slots: int) -> tuple[list[int], list[int]]:
    hi = 4 if n_slots == 3 else 3
    heights = [random.randint(2, hi) for _ in range(n_slots)]
    widths = [random.randint(2, hi) for _ in range(n_slots)]
    return heights, widths


def _line_positions(start: int, sizes: list[int]) -> list[int]:
    lines = [start]
    cur = start
    for size in sizes:
        cur += size + 1
        lines.append(cur)
    return lines


def _distance_plan() -> tuple[dict[str, int], dict[str, object]]:
    mode = random.choice(("strict", "tie"))
    if mode == "strict":
        horizontal_near = random.choice(HORIZONTAL_SIDES)
        horizontal_far = _other(horizontal_near)
        vertical_main = random.choice(VERTICAL_SIDES)
        vertical_other = _other(vertical_main)
        a, b, c = sorted(random.sample((1, 2, 3, 4), 3))
        distances = {
            horizontal_near: a,
            vertical_main: b,
            horizontal_far: c,
            vertical_other: next(v for v in (1, 2, 3, 4) if v not in {a, b, c}),
        }
        plan = {
            "mode": mode,
            "required_row_sides": [horizontal_near, horizontal_far],
            "required_col_sides": [vertical_main],
        }
        return distances, plan

    horizontal_tie = random.choice(HORIZONTAL_SIDES)
    horizontal_far = _other(horizontal_tie)
    vertical_tie = random.choice(VERTICAL_SIDES)
    vertical_other = _other(vertical_tie)
    tie_distance = random.randint(1, 3)
    far_distance = random.randint(tie_distance + 1, 4)
    other_choices = [v for v in (1, 2, 3, 4) if v != tie_distance]
    distances = {
        horizontal_tie: tie_distance,
        vertical_tie: tie_distance,
        horizontal_far: far_distance,
        vertical_other: random.choice(other_choices),
    }
    plan = {
        "mode": mode,
        "required_row_sides": [horizontal_tie, horizontal_far],
        "required_col_sides": [vertical_tie],
    }
    return distances, plan


def _sample_lane_sides(n_slots: int, plan: dict[str, object]) -> tuple[dict[int, str], dict[int, str]]:
    row_count = random.randint(2, n_slots - 1)
    col_count = random.randint(1, n_slots - 1)

    row_lanes = random.sample(range(n_slots), row_count)
    col_lanes = random.sample(range(n_slots), col_count)

    row_sides: dict[int, str] = {}
    col_sides: dict[int, str] = {}
    for lane, side in zip(row_lanes, plan["required_row_sides"]):
        row_sides[lane] = side
    for lane, side in zip(col_lanes, plan["required_col_sides"]):
        col_sides[lane] = side

    for lane in row_lanes[len(plan["required_row_sides"]) :]:
        row_sides[lane] = random.choice(HORIZONTAL_SIDES)
    for lane in col_lanes[len(plan["required_col_sides"]) :]:
        col_sides[lane] = random.choice(VERTICAL_SIDES)
    return row_sides, col_sides


def _draw_frame(
    shape: tuple[int, int],
    row_lines: list[int],
    col_lines: list[int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    r0, r1 = row_lines[0], row_lines[-1]
    c0, c1 = col_lines[0], col_lines[-1]
    for r in row_lines:
        grid[r, c0 : c1 + 1] = FRAME
    for c in col_lines:
        grid[r0 : r1 + 1, c] = FRAME
    return grid


def _bands_from_sizes(start: int, sizes: list[int]) -> list[tuple[int, int]]:
    return _bands_from_lines(_line_positions(start, sizes))


def _render_from_spec(
    grid: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    row_markers: dict[int, tuple[int, int]],
    col_markers: dict[int, tuple[int, int]],
) -> np.ndarray:
    frame = {
        "bbox": (row_lines[0], col_lines[0], row_lines[-1], col_lines[-1]),
        "row_lines": row_lines,
        "col_lines": col_lines,
        "row_bands": _bands_from_lines(row_lines),
        "col_bands": _bands_from_lines(col_lines),
    }
    return _fill_slots(grid, frame, row_markers, col_markers)


def _cross_outcomes(
    row_markers: dict[int, tuple[int, int]],
    col_markers: dict[int, tuple[int, int]],
) -> set[str]:
    outcomes: set[str] = set()
    for row_color, row_dist in row_markers.values():
        _ = row_color
        for col_color, col_dist in col_markers.values():
            _ = col_color
            if row_dist < col_dist:
                outcomes.add("horizontal")
            elif row_dist > col_dist:
                outcomes.add("vertical")
            else:
                outcomes.add("tie")
    return outcomes


def _gen_attempt() -> np.ndarray | None:
    n_slots = random.choice((3, 4))
    slot_heights, slot_widths = _sample_slot_sizes(n_slots)
    distances, plan = _distance_plan()

    row_lines = _line_positions(distances["top"] + 1, slot_heights)
    col_lines = _line_positions(distances["left"] + 1, slot_widths)
    height = row_lines[-1] + distances["bottom"] + 2
    width = col_lines[-1] + distances["right"] + 2
    if max(height, width) > 30:
        return None

    row_bands = _bands_from_lines(row_lines)
    col_bands = _bands_from_lines(col_lines)
    row_side_map, col_side_map = _sample_lane_sides(n_slots, plan)

    grid = _draw_frame((height, width), row_lines, col_lines)
    total_markers = len(row_side_map) + len(col_side_map)
    colors = random.sample(MARKER_COLORS, total_markers)

    row_markers: dict[int, tuple[int, int]] = {}
    col_markers: dict[int, tuple[int, int]] = {}

    color_index = 0
    for lane, side in row_side_map.items():
        r_lo, r_hi = row_bands[lane]
        r = random.randint(r_lo, r_hi)
        c = 0 if side == "left" else width - 1
        color = colors[color_index]
        color_index += 1
        grid[r, c] = color
        row_markers[lane] = (color, distances[side])

    for lane, side in col_side_map.items():
        c_lo, c_hi = col_bands[lane]
        c = random.randint(c_lo, c_hi)
        r = 0 if side == "top" else height - 1
        color = colors[color_index]
        color_index += 1
        grid[r, c] = color
        col_markers[lane] = (color, distances[side])

    outcomes = _cross_outcomes(row_markers, col_markers)
    if plan["mode"] == "strict" and not {"horizontal", "vertical"} <= outcomes:
        return None
    if plan["mode"] == "tie" and not {"tie", "vertical"} <= outcomes:
        return None

    expected = _render_from_spec(grid, row_lines, col_lines, row_markers, col_markers)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-ray lattice input")
