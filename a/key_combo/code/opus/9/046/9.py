from __future__ import annotations

import numpy as np


BLACK = 0
BOUNDARY_COLORS = (1, 2, 3, 4)
GREY = 5

GRID_SIZE_OPTIONS = tuple(range(10, 15))
MODE_WEIGHTS = (
    ("basic", 0.25),
    ("tie", 0.30),
    ("nonlocal", 0.25),
    ("full", 0.20),
)
MAX_GENERATION_ATTEMPTS = 1500


def _parse_markers(grid: np.ndarray) -> list[dict[str, int | str]]:
    h, w = grid.shape
    coords = np.argwhere(grid != BLACK)
    markers: list[dict[str, int | str]] = []
    seen_colors: set[int] = set()
    seen_rows: set[int] = set()
    seen_cols: set[int] = set()

    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        color = int(grid[r, c])
        assert color in BOUNDARY_COLORS, "input should contain only colored boundary markers"
        assert r in (0, h - 1) or c in (0, w - 1), "markers must lie on the boundary"
        assert not ((r in (0, h - 1)) and (c in (0, w - 1))), "markers cannot occupy corners"
        assert color not in seen_colors, "marker colors must be distinct"
        assert r not in seen_rows and c not in seen_cols, "markers must not share a row or column"
        seen_colors.add(color)
        seen_rows.add(r)
        seen_cols.add(c)

        if r == 0:
            side = "top"
        elif r == h - 1:
            side = "bottom"
        elif c == 0:
            side = "left"
        else:
            side = "right"

        markers.append({"id": len(markers), "r": r, "c": c, "color": color, "side": side})

    assert 2 <= len(markers) <= 4, "expected between two and four boundary markers"
    assert any(m["side"] in {"top", "bottom"} for m in markers), "need a vertical span marker"
    assert any(m["side"] in {"left", "right"} for m in markers), "need a horizontal span marker"
    return sorted(markers, key=lambda m: (int(m["r"]), int(m["c"])))


def _segments(length: int, lines: list[int]) -> list[tuple[int, int, int | None, int | None]]:
    ordered = sorted(int(x) for x in lines)
    segments: list[tuple[int, int, int | None, int | None]] = []
    start = 0
    prev_line: int | None = None
    for line in ordered:
        if start <= line - 1:
            segments.append((start, line - 1, prev_line, line))
        start = line + 1
        prev_line = line
    if start <= length - 1:
        segments.append((start, length - 1, prev_line, None))
    return segments


def _best_marker_for_point(
    point: tuple[int, int], markers: list[dict[str, int | str]]
) -> dict[str, int | str]:
    pr, pc = point
    return min(
        markers,
        key=lambda m: (
            abs(pr - int(m["r"])) + abs(pc - int(m["c"])),
            int(m["r"]),
            int(m["c"]),
        ),
    )


def _solve_with_metadata(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"
    markers = _parse_markers(g)
    h, w = g.shape

    vertical_markers = [m for m in markers if m["side"] in {"top", "bottom"}]
    horizontal_markers = [m for m in markers if m["side"] in {"left", "right"}]
    vertical_cols = sorted(int(m["c"]) for m in vertical_markers)
    horizontal_rows = sorted(int(m["r"]) for m in horizontal_markers)
    row_to_marker = {int(m["r"]): m for m in horizontal_markers}
    col_to_marker = {int(m["c"]): m for m in vertical_markers}

    out = np.full_like(g, BLACK)
    for marker in vertical_markers:
        out[:, int(marker["c"])] = GREY
    for marker in horizontal_markers:
        out[int(marker["r"]), :] = GREY

    regions = []
    for r0, r1, upper_row, lower_row in _segments(h, horizontal_rows):
        for c0, c1, left_col, right_col in _segments(w, vertical_cols):
            center = ((r0 + r1) // 2, (c0 + c1) // 2)
            winner = _best_marker_for_point(center, markers)
            out[r0 : r1 + 1, c0 : c1 + 1] = int(winner["color"])

            boundary_ids = set()
            if upper_row is not None:
                boundary_ids.add(int(row_to_marker[upper_row]["id"]))
            if lower_row is not None:
                boundary_ids.add(int(row_to_marker[lower_row]["id"]))
            if left_col is not None:
                boundary_ids.add(int(col_to_marker[left_col]["id"]))
            if right_col is not None:
                boundary_ids.add(int(col_to_marker[right_col]["id"]))

            dists = [
                abs(center[0] - int(m["r"])) + abs(center[1] - int(m["c"]))
                for m in markers
            ]
            best_dist = min(dists)
            tie_count = sum(dist == best_dist for dist in dists)
            regions.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "center": center,
                    "winner_id": int(winner["id"]),
                    "winner_color": int(winner["color"]),
                    "boundary_ids": boundary_ids,
                    "tie": tie_count > 1,
                }
            )

    for marker in markers:
        out[int(marker["r"]), int(marker["c"])] = int(marker["color"])

    meta = {
        "markers": markers,
        "regions": regions,
        "vertical_cols": vertical_cols,
        "horizontal_rows": horizontal_rows,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw gray spans from the boundary markers, then color each rectangular partition by the nearest marker to its center.

    Input structure:
    - A black grid contains 2-4 single-cell colored markers on the outer boundary.
    - Top/bottom markers designate full-height vertical spans through their columns.
    - Left/right markers designate full-width horizontal spans through their rows.

    Rule:
    - Extend every top/bottom marker into a full gray vertical line and every left/right marker into a full gray horizontal line.
    - Treat those gray spans as separators that partition the remaining cells into rectangles.
    - For each rectangle, compute its center using floor((min+max)/2) on rows and columns, find the boundary marker with minimum Manhattan distance to that center, and fill the rectangle with that marker's color.
    - Keep gray separator lines gray, and restore the original colored marker pixels on top of those lines.
    - If two markers are equally close to a region center, choose the one with smaller row index; if rows tie, choose the one with smaller column index.

    Key insight:
    - The same boundary markers serve two independent roles: they create the geometric partition lines and they also compete, by distance to each partition center, to decide the region's final color.
    """
    out, _ = _solve_with_metadata(grid)
    return out


def _weighted_mode(rng: np.random.Generator) -> str:
    labels = [label for label, _ in MODE_WEIGHTS]
    weights = np.array([weight for _, weight in MODE_WEIGHTS], dtype=float)
    weights /= weights.sum()
    return str(rng.choice(labels, p=weights))


def _pick_side_set(mode: str, rng: np.random.Generator) -> tuple[str, ...]:
    if mode == "tie":
        options = [
            ("top", "left"),
            ("top", "right"),
            ("bottom", "left"),
            ("bottom", "right"),
            ("top", "left", "right"),
            ("bottom", "left", "right"),
            ("top", "bottom", "left"),
            ("top", "bottom", "right"),
        ]
    elif mode == "full":
        options = [("top", "bottom", "left", "right")]
    elif mode == "nonlocal":
        options = [
            ("top", "bottom", "left"),
            ("top", "bottom", "right"),
            ("top", "left", "right"),
            ("bottom", "left", "right"),
            ("top", "bottom", "left", "right"),
        ]
    else:
        options = [
            ("top", "left"),
            ("top", "right"),
            ("bottom", "left"),
            ("bottom", "right"),
            ("top", "bottom", "left"),
            ("top", "bottom", "right"),
            ("top", "left", "right"),
            ("bottom", "left", "right"),
            ("top", "bottom", "left", "right"),
        ]
    return tuple(rng.choice(np.array(options, dtype=object)))


def _sample_positions(
    count: int,
    upper: int,
    rng: np.random.Generator,
    *,
    reserved: tuple[int, ...] = (),
    min_gap: int = 2,
) -> list[int] | None:
    candidates = list(range(1, upper - 1))
    rng.shuffle(candidates)
    picked = list(reserved)
    for value in candidates:
        if value in picked:
            continue
        if all(abs(value - other) >= min_gap for other in picked):
            picked.append(value)
        if len(picked) == len(reserved) + count:
            break
    if len(picked) != len(reserved) + count:
        return None
    return picked[len(reserved) :]


def _build_candidate(mode: str, rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.choice(GRID_SIZE_OPTIONS))
    w = int(rng.choice(GRID_SIZE_OPTIONS))
    sides = _pick_side_set(mode, rng)

    vertical_sides = [side for side in sides if side in {"top", "bottom"}]
    horizontal_sides = [side for side in sides if side in {"left", "right"}]
    markers: list[tuple[str, int]] = []

    anchor_value: int | None = None
    anchor_vertical: str | None = None
    anchor_horizontal: str | None = None
    if mode == "tie":
        anchor_vertical = vertical_sides[int(rng.integers(len(vertical_sides)))]
        anchor_horizontal = horizontal_sides[int(rng.integers(len(horizontal_sides)))]
        hi = min(h, w) - 2
        if hi <= 1:
            return None
        anchor_value = int(rng.integers(1, hi))

    vert_positions = []
    horiz_positions = []
    if anchor_value is not None:
        extra_vert = len(vertical_sides) - 1
        extra_horiz = len(horizontal_sides) - 1
        extra_vert_pos = _sample_positions(extra_vert, w, rng, reserved=(anchor_value,))
        extra_horiz_pos = _sample_positions(extra_horiz, h, rng, reserved=(anchor_value,))
        if extra_vert_pos is None or extra_horiz_pos is None:
            return None
        vert_positions = [anchor_value] + extra_vert_pos
        horiz_positions = [anchor_value] + extra_horiz_pos
    else:
        vert_positions = _sample_positions(len(vertical_sides), w, rng)
        horiz_positions = _sample_positions(len(horizontal_sides), h, rng)
        if vert_positions is None or horiz_positions is None:
            return None

    vert_map: dict[str, int] = {}
    horiz_map: dict[str, int] = {}
    vi = 0
    hi = 0
    for side in vertical_sides:
        if side == anchor_vertical:
            vert_map[side] = int(anchor_value)
        else:
            vert_map[side] = int(vert_positions[1 + vi] if anchor_value is not None else vert_positions[vi])
            vi += 1
    for side in horizontal_sides:
        if side == anchor_horizontal:
            horiz_map[side] = int(anchor_value)
        else:
            horiz_map[side] = int(horiz_positions[1 + hi] if anchor_value is not None else horiz_positions[hi])
            hi += 1

    side_order = list(sides)
    rng.shuffle(side_order)
    colors = [int(x) for x in rng.choice(np.array(BOUNDARY_COLORS), size=len(side_order), replace=False)]

    grid = np.zeros((h, w), dtype=int)
    for color, side in zip(colors, side_order):
        if side == "top":
            r, c = 0, vert_map[side]
        elif side == "bottom":
            r, c = h - 1, vert_map[side]
        elif side == "left":
            r, c = horiz_map[side], 0
        else:
            r, c = horiz_map[side], w - 1
        grid[r, c] = color
        markers.append((side, color))

    return grid


def _cellwise_conflict(region: dict[str, object], markers: list[dict[str, int | str]]) -> bool:
    r0, c0, r1, c1 = region["bbox"]
    winner_id = int(region["winner_id"])
    for r in range(int(r0), int(r1) + 1):
        for c in range(int(c0), int(c1) + 1):
            best = _best_marker_for_point((r, c), markers)
            if int(best["id"]) != winner_id:
                return True
    return False


def _analyze_candidate(inp: np.ndarray) -> dict[str, object]:
    out, meta = _solve_with_metadata(inp)
    markers = meta["markers"]
    regions = meta["regions"]

    region_colors = {int(region["winner_color"]) for region in regions}
    tie_regions = sum(bool(region["tie"]) for region in regions)
    nonlocal_regions = sum(int(region["winner_id"]) not in region["boundary_ids"] for region in regions)
    conflict_regions = sum(_cellwise_conflict(region, markers) for region in regions)

    h, w = inp.shape
    grey_cells = int(np.sum(out == GREY))
    marker_cells = len(markers)
    analysis = {
        "region_count": len(regions),
        "region_color_count": len(region_colors),
        "tie_regions": tie_regions,
        "nonlocal_regions": nonlocal_regions,
        "conflict_regions": conflict_regions,
        "grey_fraction": grey_cells / float(h * w),
        "marker_count": marker_cells,
        "has_top_bottom": any(m["side"] == "top" for m in markers) and any(m["side"] == "bottom" for m in markers),
        "has_left_right": any(m["side"] == "left" for m in markers) and any(m["side"] == "right" for m in markers),
        "out": out,
    }
    return analysis


def _accept_candidate(mode: str, analysis: dict[str, object]) -> bool:
    if int(analysis["region_count"]) < 4:
        return False
    if int(analysis["region_color_count"]) < 2:
        return False
    if int(analysis["conflict_regions"]) < 1:
        return False
    if float(analysis["grey_fraction"]) > 0.42:
        return False

    if mode == "tie":
        return int(analysis["tie_regions"]) >= 1
    if mode == "nonlocal":
        return int(analysis["nonlocal_regions"]) >= 1
    if mode == "full":
        return (
            int(analysis["marker_count"]) == 4
            and bool(analysis["has_top_bottom"])
            and bool(analysis["has_left_right"])
            and int(analysis["nonlocal_regions"]) >= 1
        )
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        mode = _weighted_mode(rng)
        inp = _build_candidate(mode, rng)
        if inp is None:
            continue
        analysis = _analyze_candidate(inp)
        if _accept_candidate(mode, analysis):
            solved = main(inp)
            assert np.array_equal(solved, analysis["out"])
            return inp
    raise RuntimeError("failed to generate a valid boundary-marker partition puzzle")
