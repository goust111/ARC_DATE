from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def _horizontal_segments(grid: np.ndarray) -> list[dict]:
    segments: list[dict] = []
    seg_id = 0
    height, width = grid.shape
    for row in range(height):
        col = 0
        while col < width:
            color = int(grid[row, col])
            if color == 0:
                col += 1
                continue
            start = col
            while col + 1 < width and int(grid[row, col + 1]) == color:
                col += 1
            end = col
            if end - start + 1 >= 2:
                segments.append(
                    {
                        "id": seg_id,
                        "orientation": "h",
                        "color": color,
                        "line": row,
                        "start": start,
                        "end": end,
                    }
                )
                seg_id += 1
            col += 1
    return segments


def _vertical_segments(grid: np.ndarray) -> list[dict]:
    segments: list[dict] = []
    seg_id = 0
    height, width = grid.shape
    for col in range(width):
        row = 0
        while row < height:
            color = int(grid[row, col])
            if color == 0:
                row += 1
                continue
            start = row
            while row + 1 < height and int(grid[row + 1, col]) == color:
                row += 1
            end = row
            if end - start + 1 >= 2:
                segments.append(
                    {
                        "id": seg_id,
                        "orientation": "v",
                        "color": color,
                        "line": col,
                        "start": start,
                        "end": end,
                    }
                )
                seg_id += 1
            row += 1
    return segments


def _covers_vertical(segments: list[dict], row: int) -> bool:
    return any(int(seg["start"]) <= row <= int(seg["end"]) for seg in segments)


def _covers_horizontal(segments: list[dict], col: int) -> bool:
    return any(int(seg["start"]) <= col <= int(seg["end"]) for seg in segments)


def _extract_rectangles(segments: list[dict]) -> list[tuple[int, int, int, int]]:
    horizontals = [seg for seg in segments if seg["orientation"] == "h"]
    verticals = [seg for seg in segments if seg["orientation"] == "v"]

    h_by_span: dict[tuple[int, int], list[dict]] = defaultdict(list)
    v_by_span: dict[tuple[int, int], list[dict]] = defaultdict(list)
    v_by_col: dict[int, list[dict]] = defaultdict(list)
    h_by_row: dict[int, list[dict]] = defaultdict(list)

    for seg in horizontals:
        h_by_span[(int(seg["start"]), int(seg["end"]))].append(seg)
        h_by_row[int(seg["line"])].append(seg)
    for seg in verticals:
        v_by_span[(int(seg["start"]), int(seg["end"]))].append(seg)
        v_by_col[int(seg["line"])].append(seg)

    rectangles: set[tuple[int, int, int, int]] = set()

    for (c0, c1), segs in h_by_span.items():
        rows = sorted({int(seg["line"]) for seg in segs})
        if len(rows) < 2:
            continue
        for i, r0 in enumerate(rows):
            for r1 in rows[i + 1 :]:
                if r1 - r0 < 4:
                    continue
                left_side = v_by_col.get(c0, [])
                right_side = v_by_col.get(c1, [])
                if not left_side or not right_side:
                    continue
                if not _covers_vertical(left_side, r0):
                    continue
                if not _covers_vertical(left_side, r1):
                    continue
                if not _covers_vertical(right_side, r0):
                    continue
                if not _covers_vertical(right_side, r1):
                    continue
                rectangles.add((r0, c0, r1, c1))

    for (r0, r1), segs in v_by_span.items():
        cols = sorted({int(seg["line"]) for seg in segs})
        if len(cols) < 2:
            continue
        for i, c0 in enumerate(cols):
            for c1 in cols[i + 1 :]:
                if c1 - c0 < 4:
                    continue
                top_side = h_by_row.get(r0, [])
                bottom_side = h_by_row.get(r1, [])
                if not top_side or not bottom_side:
                    continue
                if not _covers_horizontal(top_side, c0):
                    continue
                if not _covers_horizontal(top_side, c1):
                    continue
                if not _covers_horizontal(bottom_side, c0):
                    continue
                if not _covers_horizontal(bottom_side, c1):
                    continue
                rectangles.add((r0, c0, r1, c1))

    return sorted(rectangles)


def _majority_color(colors: list[int]) -> int:
    counts = Counter(colors)
    best = max(counts.values())
    winners = {color for color, count in counts.items() if count == best}
    for color in colors:
        if color in winners:
            return int(color)
    raise AssertionError("empty color list")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each broken rectangle frame from its aligned line fragments and recolor it by fragment majority.

    Input structure:
    - Black background with several horizontal or vertical monochrome line segments of length at least 2.
    - The segments are fragments taken from one-pixel-thick hollow rectangle borders, and one rectangle may use multiple fragment colors.
    - Fragments from the same rectangle stay aligned on the original rectangle rows and columns.

    Rule:
    - Group fragments into rectangles by matching shared rectangle endpoints: horizontal fragments imply a left and right side column, while vertical fragments imply a top and bottom side row.
    - For each group, recover the full hollow rectangle spanning its extreme side rows and columns and discard the original partial fragments.
    - Color the rebuilt border with the fragment color that appears on the most fragments in that group.

    Key insight:
    - The line pieces must first be organized by their hidden common rectangle geometry; only after recovering that grouping can the majority fragment color be applied to the completed frame.
    """
    g = np.asarray(grid, dtype=int)
    segments = _horizontal_segments(g) + _vertical_segments(g)
    rectangles = _extract_rectangles(segments)

    memberships: dict[tuple[int, int, int, int], list[dict]] = {bbox: [] for bbox in rectangles}
    for seg in segments:
        matched: list[tuple[int, int, int, int]] = []
        for bbox in rectangles:
            r0, c0, r1, c1 = bbox
            if seg["orientation"] == "h":
                if int(seg["line"]) in (r0, r1) and c0 <= int(seg["start"]) and int(seg["end"]) <= c1:
                    matched.append(bbox)
            else:
                if int(seg["line"]) in (c0, c1) and r0 <= int(seg["start"]) and int(seg["end"]) <= r1:
                    matched.append(bbox)
        if matched:
            assert len(matched) == 1, f"ambiguous fragment assignment: {seg} -> {matched}"
            memberships[matched[0]].append(seg)

    assert rectangles, "expected at least one rectangle"
    assert all(memberships[bbox] for bbox in rectangles), "each rectangle needs visible fragments"
    covered = sum(len(items) for items in memberships.values())
    assert covered == len(segments), "every visible fragment must belong to exactly one rectangle"

    out = np.zeros_like(g)
    for r0, c0, r1, c1 in rectangles:
        color = _majority_color([int(seg["color"]) for seg in memberships[(r0, c0, r1, c1)]])
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


# --------------- generator (codex) ---------------

COLORS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
MODES = ("h", "v")
RECT_COUNT_RANGE = (1, 4)
SIDE_LENGTH_RANGE = (8, 13)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 400


def _paint_h(grid: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    grid[row, c0 : c1 + 1] = int(color)


def _paint_v(grid: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    grid[r0 : r1 + 1, col] = int(color)


def _vertical_side_runs(r0: int, r1: int, major: int, minor: int, pattern: str) -> list[tuple[int, int, int]]:
    if pattern == "full":
        return [(r0, r1, major)]
    if pattern == "split":
        return [(r0, r0 + 1, major), (r1 - 1, r1, major)]
    if pattern == "minor":
        height = r1 - r0 + 1
        assert height >= 8
        mid_start = r0 + 3 + (height - 8) // 2
        return [(r0, r0 + 1, major), (mid_start, mid_start + 1, minor), (r1 - 1, r1, major)]
    raise AssertionError(f"unknown vertical pattern: {pattern}")


def _horizontal_side_runs(c0: int, c1: int, major: int, minor: int, pattern: str) -> list[tuple[int, int, int]]:
    if pattern == "full":
        return [(c0, c1, major)]
    if pattern == "split":
        return [(c0, c0 + 1, major), (c1 - 1, c1, major)]
    if pattern == "minor":
        width = c1 - c0 + 1
        assert width >= 8
        mid_start = c0 + 3 + (width - 8) // 2
        return [(c0, c0 + 1, major), (mid_start, mid_start + 1, minor), (c1 - 1, c1, major)]
    raise AssertionError(f"unknown horizontal pattern: {pattern}")


def _choose_side_patterns(can_use_minor: bool, rng: np.random.Generator) -> tuple[str, str]:
    basic = ["full", "split"]
    patterns = list(basic)
    if can_use_minor:
        patterns.append("minor")

    while True:
        left = str(rng.choice(patterns))
        right = str(rng.choice(patterns))
        if can_use_minor and left == "minor" and right == "minor":
            return left, right
        if can_use_minor and (left == "minor" or right == "minor"):
            return left, right
        if not can_use_minor:
            return left, right
        if rng.random() < 0.25:
            return left, right


def _can_use_minor(spec: dict) -> bool:
    r0, c0, r1, c1 = map(int, spec["bbox"])
    if str(spec["mode"]) == "h":
        return (r1 - r0 + 1) >= 8
    return (c1 - c0 + 1) >= 8


def _assign_patterns(spec: dict, rng: np.random.Generator, force_minor: bool = False) -> None:
    can_use_minor = _can_use_minor(spec)
    if str(spec["mode"]) == "h":
        if force_minor and can_use_minor:
            first = "minor"
            second = str(rng.choice(["full", "split"]))
        else:
            first, second = _choose_side_patterns(can_use_minor, rng)
    else:
        if force_minor and can_use_minor:
            first = "minor"
            second = str(rng.choice(["full", "split"]))
        else:
            first, second = _choose_side_patterns(can_use_minor, rng)
    spec["patterns"] = (first, second)
    spec["used_minor"] = "minor" in spec["patterns"]


def _bbox_free(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_rectangle_input(grid: np.ndarray, spec: dict) -> None:
    r0, c0, r1, c1 = map(int, spec["bbox"])
    major = int(spec["major"])
    minor = int(spec["minor"])
    mode = str(spec["mode"])

    if mode == "h":
        _paint_h(grid, r0, c0, c1, major)
        _paint_h(grid, r1, c0, c1, major)
        left_pattern, right_pattern = spec["patterns"]
        for start, end, color in _vertical_side_runs(r0, r1, major, minor, left_pattern):
            _paint_v(grid, c0, start, end, color)
        for start, end, color in _vertical_side_runs(r0, r1, major, minor, right_pattern):
            _paint_v(grid, c1, start, end, color)
    else:
        _paint_v(grid, c0, r0, r1, major)
        _paint_v(grid, c1, r0, r1, major)
        top_pattern, bottom_pattern = spec["patterns"]
        for start, end, color in _horizontal_side_runs(c0, c1, major, minor, top_pattern):
            _paint_h(grid, r0, start, end, color)
        for start, end, color in _horizontal_side_runs(c0, c1, major, minor, bottom_pattern):
            _paint_h(grid, r1, start, end, color)


def _draw_rectangle_output(grid: np.ndarray, spec: dict) -> None:
    r0, c0, r1, c1 = map(int, spec["bbox"])
    color = int(spec["major"])
    _paint_h(grid, r0, c0, c1, color)
    _paint_h(grid, r1, c0, c1, color)
    _paint_v(grid, c0, r0, r1, color)
    _paint_v(grid, c1, r0, r1, color)


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    rect_count = int(rng.integers(*RECT_COUNT_RANGE))
    min_dim = 14 + 5 * (rect_count - 1)
    height = int(rng.integers(min_dim, 31))
    width = int(rng.integers(min_dim, 31))

    mask = np.zeros((height, width), dtype=bool)
    specs: list[dict] = []
    used_h_spans: set[tuple[int, int]] = set()
    used_v_spans: set[tuple[int, int]] = set()

    for _ in range(rect_count):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            box_h = int(rng.integers(*SIDE_LENGTH_RANGE))
            box_w = int(rng.integers(*SIDE_LENGTH_RANGE))
            if box_h + 2 > height or box_w + 2 > width:
                continue
            r0 = int(rng.integers(1, height - box_h))
            c0 = int(rng.integers(1, width - box_w))
            r1 = r0 + box_h - 1
            c1 = c0 + box_w - 1
            bbox = (r0, c0, r1, c1)
            if (c0, c1) in used_h_spans or (r0, r1) in used_v_spans:
                continue
            if not _bbox_free(mask, bbox, pad=1):
                continue

            major = int(rng.choice(COLORS))
            minor_pool = COLORS[COLORS != major]
            minor = int(rng.choice(minor_pool))
            mode = str(rng.choice(MODES))
            specs.append(
                {
                    "bbox": bbox,
                    "major": major,
                    "minor": minor,
                    "mode": mode,
                }
            )
            used_h_spans.add((c0, c1))
            used_v_spans.add((r0, r1))
            _mark_bbox(mask, bbox, pad=1)
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)
    for spec in specs:
        _assign_patterns(spec, rng)
    if not any(bool(spec["used_minor"]) for spec in specs):
        for spec in specs:
            if _can_use_minor(spec):
                _assign_patterns(spec, rng, force_minor=True)
                break
    for spec in specs:
        _draw_rectangle_input(grid, spec)
        _draw_rectangle_output(out, spec)

    return grid, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        sample = _gen_attempt(rng)
        if sample is None:
            continue
        grid, _out = sample
        return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
