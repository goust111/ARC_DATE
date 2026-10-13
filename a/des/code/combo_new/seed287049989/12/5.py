import random

import numpy as np


BACKGROUND = 8
PALETTE = (0, 1, 2, 3, 4, 5, 6, 7, 9)
GEN_ATTEMPTS = 400
SCENARIOS = ("single", "tie", "cross", "mixed")


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _is_boundary_cell(row: int, col: int, height: int, width: int) -> bool:
    return row in (0, height - 1) or col in (0, width - 1)


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict[str, int]], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = _background_color(g)

    lines: list[dict[str, int]] = []
    rectangles: list[dict[str, int]] = []

    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        coords = np.argwhere(g == color)
        if coords.shape[0] == 2:
            p0 = tuple(map(int, coords[0]))
            p1 = tuple(map(int, coords[1]))
            if _is_boundary_cell(*p0, height, width) and _is_boundary_cell(*p1, height, width):
                if {p0[0], p1[0]} == {0, height - 1} and p0[1] == p1[1]:
                    lines.append({"orientation": "V", "index": p0[1], "color": color})
                    continue
                if {p0[1], p1[1]} == {0, width - 1} and p0[0] == p1[0]:
                    lines.append({"orientation": "H", "index": p0[0], "color": color})
                    continue

        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert int(coords.shape[0]) == expected_area, "foreground objects must be solid rectangles"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be monochrome"
        rectangles.append({"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1})

    assert 1 <= len(lines) <= 3, "expected one to three marker pairs"
    seen_lines = set()
    for line in lines:
        key = (line["orientation"], line["index"])
        assert key not in seen_lines, "marker pairs must define distinct cut lines"
        seen_lines.add(key)

    assert rectangles, "expected at least one rectangle"
    return bg, lines, rectangles


def _segments(lo: int, hi: int, cuts: set[int]) -> list[tuple[int, int]]:
    inner_cuts = sorted(cut for cut in cuts if lo < cut < hi)
    segments: list[tuple[int, int]] = []
    start = lo
    for cut in inner_cuts:
        if start <= cut - 1:
            segments.append((start, cut - 1))
        start = cut + 1
    if start <= hi:
        segments.append((start, hi))
    return segments


def _rectangle_fragments(
    rect: dict[str, int], cut_rows: set[int], cut_cols: set[int], grid_shape: tuple[int, int]
) -> list[dict[str, float]]:
    height, width = map(int, grid_shape)
    center_row = (height - 1) / 2.0
    center_col = (width - 1) / 2.0

    row_segments = _segments(rect["r0"], rect["r1"], cut_rows)
    col_segments = _segments(rect["c0"], rect["c1"], cut_cols)
    fragments: list[dict[str, float]] = []
    for r0, r1 in row_segments:
        for c0, c1 in col_segments:
            frag_center_row = (r0 + r1) / 2.0
            frag_center_col = (c0 + c1) / 2.0
            row_gap = abs(frag_center_row - center_row)
            col_gap = abs(frag_center_col - center_col)
            fragments.append(
                {
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                    "dist2": row_gap * row_gap + col_gap * col_gap,
                    "row_gap": row_gap,
                    "col_gap": col_gap,
                }
            )
    assert fragments, "every rectangle must leave at least one fragment"
    return fragments


def _best_fragment(fragments: list[dict[str, float]]) -> dict[str, float]:
    return min(
        fragments,
        key=lambda frag: (
            -int(frag["area"]),
            float(frag["dist2"]),
            float(frag["row_gap"]),
            float(frag["col_gap"]),
            int(frag["r0"]),
            int(frag["c0"]),
            int(frag["r1"]),
            int(frag["c1"]),
        ),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clear the paired edge markers, cut each solid rectangle along the implied full-row/full-column lines, and keep only its best fragment.

    Input structure:
    - The dominant color is the azure background.
    - Every other large color forms one solid axis-aligned rectangle, with rectangle colors unique and non-overlapping.
    - One to three extra colors appear exactly twice as single boundary markers, and each same-color pair sits on opposite sides in one shared row or one shared column.

    Rule:
    - Each marker pair defines an invisible horizontal or vertical cut line spanning the whole grid.
    - A line cuts a rectangle only when it passes through that rectangle's interior; multiple lines split the same rectangle into multiple rectangular fragments.
    - For each original rectangle, keep the fragment with maximum area and erase the others to background; when several fragments tie in area, keep the one whose fragment-center is closest to the grid center.

    Key insight:
    - The boundary pixels are not part of the output objects: they are paired instructions that specify hidden slicing lines, after which each rectangle is reduced to its largest surviving piece.
    """
    g = np.asarray(grid, dtype=int)
    bg, lines, rectangles = _extract_scene(g)
    cut_rows = {line["index"] for line in lines if line["orientation"] == "H"}
    cut_cols = {line["index"] for line in lines if line["orientation"] == "V"}

    out = np.full_like(g, bg)
    for rect in rectangles:
        fragments = _rectangle_fragments(rect, cut_rows, cut_cols, g.shape)
        keep = _best_fragment(fragments)
        out[int(keep["r0"]) : int(keep["r1"]) + 1, int(keep["c0"]) : int(keep["c1"]) + 1] = rect["color"]
    return out


def _can_place_bbox(occupied: np.ndarray, bbox: tuple[int, int, int, int], gap: int = 1) -> bool:
    height, width = map(int, occupied.shape)
    r0, c0, r1, c1 = bbox
    if r0 < 1 or c0 < 1 or r1 > height - 2 or c1 > width - 2:
        return False
    rr0 = max(1, r0 - gap)
    cc0 = max(1, c0 - gap)
    rr1 = min(height - 2, r1 + gap)
    cc1 = min(width - 2, c1 + gap)
    return not np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _occupy_bbox(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, int]:
    r0, c0, r1, c1 = bbox
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True
    return {"r0": r0, "c0": c0, "r1": r1, "c1": c1}


def _touches_forbidden(
    bbox: tuple[int, int, int, int], forbid_rows: tuple[int, ...], forbid_cols: tuple[int, ...]
) -> bool:
    r0, c0, r1, c1 = bbox
    return any(r0 <= row <= r1 for row in forbid_rows) or any(c0 <= col <= c1 for col in forbid_cols)


def _side_pairs(cap_a: int, cap_b: int, tie: bool) -> list[tuple[int, int]]:
    pairs = []
    for a in range(2, min(5, cap_a) + 1):
        for b in range(2, min(5, cap_b) + 1):
            if tie and a == b:
                pairs.append((a, b))
            if not tie and a != b:
                pairs.append((a, b))
    return pairs


def _place_cut_rect(
    occupied: np.ndarray,
    orientation: str,
    line_index: int,
    *,
    tie: bool,
    forbid_rows: tuple[int, ...] = (),
    forbid_cols: tuple[int, ...] = (),
) -> dict[str, int] | None:
    height, width = map(int, occupied.shape)
    max_rect_height = min(8, height - 2)
    max_rect_width = min(8, width - 2)
    if max_rect_height < 4 or max_rect_width < 4:
        return None

    for _ in range(250):
        if orientation == "V":
            pairs = _side_pairs(line_index - 1, width - 2 - line_index, tie)
            if not pairs:
                return None
            left, right = random.choice(pairs)
            rect_height = random.randint(4, max_rect_height)
            if height - rect_height - 1 < 1:
                continue
            r0 = random.randint(1, height - rect_height - 1)
            bbox = (r0, line_index - left, r0 + rect_height - 1, line_index + right)
        else:
            pairs = _side_pairs(line_index - 1, height - 2 - line_index, tie)
            if not pairs:
                return None
            top, bottom = random.choice(pairs)
            rect_width = random.randint(4, max_rect_width)
            if width - rect_width - 1 < 1:
                continue
            c0 = random.randint(1, width - rect_width - 1)
            bbox = (line_index - top, c0, line_index + bottom, c0 + rect_width - 1)

        if _touches_forbidden(bbox, forbid_rows, forbid_cols):
            continue
        if _can_place_bbox(occupied, bbox):
            return _occupy_bbox(occupied, bbox)
    return None


def _place_cross_rect(occupied: np.ndarray, row: int, col: int) -> dict[str, int] | None:
    height, width = map(int, occupied.shape)
    vertical_pairs = _side_pairs(col - 1, width - 2 - col, tie=False)
    horizontal_pairs = _side_pairs(row - 1, height - 2 - row, tie=False)
    if not vertical_pairs or not horizontal_pairs:
        return None

    for _ in range(250):
        left, right = random.choice(vertical_pairs)
        top, bottom = random.choice(horizontal_pairs)
        bbox = (row - top, col - left, row + bottom, col + right)
        if _can_place_bbox(occupied, bbox):
            return _occupy_bbox(occupied, bbox)
    return None


def _place_uncut_rect(
    occupied: np.ndarray, *, forbid_rows: tuple[int, ...] = (), forbid_cols: tuple[int, ...] = ()
) -> dict[str, int] | None:
    height, width = map(int, occupied.shape)
    max_rect_height = min(8, height - 2)
    max_rect_width = min(8, width - 2)
    if max_rect_height < 4 or max_rect_width < 4:
        return None

    for _ in range(250):
        rect_height = random.randint(4, max_rect_height)
        rect_width = random.randint(4, max_rect_width)
        if height - rect_height - 1 < 1 or width - rect_width - 1 < 1:
            continue
        r0 = random.randint(1, height - rect_height - 1)
        c0 = random.randint(1, width - rect_width - 1)
        bbox = (r0, c0, r0 + rect_height - 1, c0 + rect_width - 1)
        if _touches_forbidden(bbox, forbid_rows, forbid_cols):
            continue
        if _can_place_bbox(occupied, bbox):
            return _occupy_bbox(occupied, bbox)
    return None


def _spaced_positions(length: int, count: int, *, margin: int) -> list[int] | None:
    candidates = list(range(margin, length - margin))
    random.shuffle(candidates)
    chosen: list[int] = []
    for value in candidates:
        if all(abs(value - prev) >= 4 for prev in chosen):
            chosen.append(value)
            if len(chosen) == count:
                return chosen
    return None


def _render_scene(height: int, width: int, rects: list[dict[str, int]], lines: list[dict[str, int]]) -> np.ndarray:
    needed = len(rects) + len(lines)
    assert needed <= len(PALETTE), "scene uses more unique colors than the palette allows"
    colors = random.sample(PALETTE, needed)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for rect, color in zip(rects, colors[: len(rects)]):
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = color

    for line, color in zip(lines, colors[len(rects) :]):
        if line["orientation"] == "V":
            grid[0, line["index"]] = color
            grid[height - 1, line["index"]] = color
        else:
            grid[line["index"], 0] = color
            grid[line["index"], width - 1] = color
    return grid


def _find_idle_line(
    rects: list[dict[str, int]], height: int, width: int, existing_lines: list[dict[str, int]]
) -> dict[str, int] | None:
    used_rows = {line["index"] for line in existing_lines if line["orientation"] == "H"}
    used_cols = {line["index"] for line in existing_lines if line["orientation"] == "V"}
    candidates: list[dict[str, int]] = []

    for row in range(2, height - 2):
        if row in used_rows:
            continue
        if all(not (rect["r0"] <= row <= rect["r1"]) for rect in rects):
            candidates.append({"orientation": "H", "index": row})
    for col in range(2, width - 2):
        if col in used_cols:
            continue
        if all(not (rect["c0"] <= col <= rect["c1"]) for rect in rects):
            candidates.append({"orientation": "V", "index": col})

    if not candidates:
        return None
    return random.choice(candidates)


def _scene_features(grid: np.ndarray) -> dict[str, int | bool]:
    _bg, lines, rects = _extract_scene(grid)
    cut_rows = {line["index"] for line in lines if line["orientation"] == "H"}
    cut_cols = {line["index"] for line in lines if line["orientation"] == "V"}
    line_hits = [0] * len(lines)
    has_tie = False
    has_multi_cut = False
    affected_rects = 0

    for rect in rects:
        hit_count = 0
        for idx, line in enumerate(lines):
            if line["orientation"] == "V":
                hit = rect["c0"] < line["index"] < rect["c1"]
            else:
                hit = rect["r0"] < line["index"] < rect["r1"]
            if hit:
                hit_count += 1
                line_hits[idx] += 1
        fragments = _rectangle_fragments(rect, cut_rows, cut_cols, grid.shape)
        max_area = max(int(fragment["area"]) for fragment in fragments)
        if sum(int(fragment["area"]) == max_area for fragment in fragments) > 1:
            has_tie = True
        if hit_count > 0:
            affected_rects += 1
        if hit_count > 1:
            has_multi_cut = True

    return {
        "line_count": len(lines),
        "both_orientations": len({line["orientation"] for line in lines}) > 1,
        "idle_line": any(hit == 0 for hit in line_hits),
        "has_tie": has_tie,
        "has_multi_cut": has_multi_cut,
        "affected_rects": affected_rects,
    }


def _build_single() -> np.ndarray | None:
    for _ in range(120):
        height = random.randint(12, 26)
        width = random.randint(12, 26)
        orientation = random.choice(("V", "H"))
        positions = _spaced_positions(width if orientation == "V" else height, 1, margin=4)
        if positions is None:
            continue
        line_index = positions[0]
        occupied = np.zeros((height, width), dtype=bool)
        rects: list[dict[str, int]] = []

        main_rect = _place_cut_rect(occupied, orientation, line_index, tie=False)
        if main_rect is None:
            continue
        rects.append(main_rect)

        for _ in range(random.randint(1, 3)):
            if random.random() < 0.6:
                extra = _place_cut_rect(occupied, orientation, line_index, tie=False)
            else:
                extra = _place_uncut_rect(
                    occupied,
                    forbid_rows=(line_index,) if orientation == "H" else (),
                    forbid_cols=(line_index,) if orientation == "V" else (),
                )
            if extra is not None:
                rects.append(extra)

        if len(rects) < 2:
            extra = _place_uncut_rect(
                occupied,
                forbid_rows=(line_index,) if orientation == "H" else (),
                forbid_cols=(line_index,) if orientation == "V" else (),
            )
            if extra is not None:
                rects.append(extra)
        if len(rects) < 2:
            continue

        return _render_scene(height, width, rects, [{"orientation": orientation, "index": line_index}])
    return None


def _build_tie() -> np.ndarray | None:
    for _ in range(120):
        height = random.randint(12, 26)
        width = random.randint(12, 26)
        orientation = random.choice(("V", "H"))
        center = (width - 1) // 2 if orientation == "V" else (height - 1) // 2
        raw_positions = _spaced_positions(width if orientation == "V" else height, 1, margin=4)
        if raw_positions is None:
            continue
        line_index = raw_positions[0]
        if (width if orientation == "V" else height) % 2 == 1 and line_index == center:
            alternatives = [
                pos
                for pos in range(4, (width if orientation == "V" else height) - 4)
                if pos != center
            ]
            if not alternatives:
                continue
            line_index = random.choice(alternatives)

        occupied = np.zeros((height, width), dtype=bool)
        rects: list[dict[str, int]] = []
        tie_rect = _place_cut_rect(occupied, orientation, line_index, tie=True)
        if tie_rect is None:
            continue
        rects.append(tie_rect)

        for _ in range(random.randint(1, 2)):
            if random.random() < 0.5:
                extra = _place_cut_rect(occupied, orientation, line_index, tie=False)
            else:
                extra = _place_uncut_rect(
                    occupied,
                    forbid_rows=(line_index,) if orientation == "H" else (),
                    forbid_cols=(line_index,) if orientation == "V" else (),
                )
            if extra is not None:
                rects.append(extra)

        if len(rects) < 2:
            extra = _place_uncut_rect(
                occupied,
                forbid_rows=(line_index,) if orientation == "H" else (),
                forbid_cols=(line_index,) if orientation == "V" else (),
            )
            if extra is not None:
                rects.append(extra)
        if len(rects) < 2:
            continue

        return _render_scene(height, width, rects, [{"orientation": orientation, "index": line_index}])
    return None


def _build_cross() -> np.ndarray | None:
    for _ in range(120):
        height = random.randint(18, 28)
        width = random.randint(18, 28)
        row_positions = _spaced_positions(height, 1, margin=5)
        col_positions = _spaced_positions(width, 1, margin=5)
        if row_positions is None or col_positions is None:
            continue
        row = row_positions[0]
        col = col_positions[0]

        occupied = np.zeros((height, width), dtype=bool)
        rects: list[dict[str, int]] = []
        cross_rect = _place_cross_rect(occupied, row, col)
        if cross_rect is None:
            continue
        rects.append(cross_rect)

        for _ in range(random.randint(1, 2)):
            kind = random.choice(("V", "H", "U"))
            if kind == "V":
                extra = _place_cut_rect(occupied, "V", col, tie=False, forbid_rows=(row,))
            elif kind == "H":
                extra = _place_cut_rect(occupied, "H", row, tie=False, forbid_cols=(col,))
            else:
                extra = _place_uncut_rect(occupied, forbid_rows=(row,), forbid_cols=(col,))
            if extra is not None:
                rects.append(extra)

        if len(rects) < 2:
            if random.random() < 0.5:
                extra = _place_cut_rect(occupied, "V", col, tie=False, forbid_rows=(row,))
            else:
                extra = _place_cut_rect(occupied, "H", row, tie=False, forbid_cols=(col,))
            if extra is not None:
                rects.append(extra)
        if len(rects) < 2:
            continue

        lines = [{"orientation": "V", "index": col}, {"orientation": "H", "index": row}]
        return _render_scene(height, width, rects, lines)
    return None


def _build_mixed() -> np.ndarray | None:
    for _ in range(120):
        height = random.randint(18, 28)
        width = random.randint(18, 28)
        row_positions = _spaced_positions(height, 1, margin=5)
        col_positions = _spaced_positions(width, 1, margin=5)
        if row_positions is None or col_positions is None:
            continue
        row = row_positions[0]
        col = col_positions[0]

        occupied = np.zeros((height, width), dtype=bool)
        rects: list[dict[str, int]] = []
        cross_rect = _place_cross_rect(occupied, row, col)
        if cross_rect is None:
            continue
        rects.append(cross_rect)

        extra_kind = random.choice(("V", "H"))
        if extra_kind == "V":
            extra = _place_cut_rect(occupied, "V", col, tie=False, forbid_rows=(row,))
        else:
            extra = _place_cut_rect(occupied, "H", row, tie=False, forbid_cols=(col,))
        if extra is None:
            continue
        rects.append(extra)

        filler = _place_uncut_rect(occupied, forbid_rows=(row,), forbid_cols=(col,))
        if filler is not None:
            rects.append(filler)

        base_lines = [{"orientation": "V", "index": col}, {"orientation": "H", "index": row}]
        idle = _find_idle_line(rects, height, width, base_lines)
        if idle is None:
            continue
        lines = base_lines + [idle]
        return _render_scene(height, width, rects, lines)
    return None


def _build_scene(scenario: str) -> np.ndarray | None:
    if scenario == "single":
        return _build_single()
    if scenario == "tie":
        return _build_tie()
    if scenario == "cross":
        return _build_cross()
    if scenario == "mixed":
        return _build_mixed()
    raise ValueError(f"unknown scenario: {scenario}")


def _scenario_ok(features: dict[str, int | bool], scenario: str) -> bool:
    if scenario == "single":
        return (
            features["line_count"] == 1
            and int(features["affected_rects"]) >= 1
            and not bool(features["has_tie"])
            and not bool(features["has_multi_cut"])
        )
    if scenario == "tie":
        return features["line_count"] == 1 and bool(features["has_tie"])
    if scenario == "cross":
        return (
            features["line_count"] == 2
            and bool(features["both_orientations"])
            and bool(features["has_multi_cut"])
            and not bool(features["idle_line"])
        )
    if scenario == "mixed":
        return (
            features["line_count"] == 3
            and bool(features["both_orientations"])
            and bool(features["has_multi_cut"])
            and bool(features["idle_line"])
        )
    return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        scenario = random.choice(SCENARIOS)
        grid = _build_scene(scenario)
        if grid is None:
            continue
        try:
            main(grid)
            features = _scene_features(grid)
        except AssertionError:
            continue
        if _scenario_ok(features, scenario):
            return grid
    raise RuntimeError("failed to generate a valid cutting-line ARC input")
