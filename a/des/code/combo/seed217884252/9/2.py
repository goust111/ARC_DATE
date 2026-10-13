import random

import numpy as np


def _crop_non_black(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != 0)
    assert coords.size > 0, "expected at least one non-black cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _segment_sort_key(segment: dict) -> tuple[int, int, int, int]:
    if segment["kind"] == "h":
        top = int(segment["row"])
        left = int(segment["c0"])
        span = int(segment["c1"] - segment["c0"] + 1)
        return (top, left, 0, span)
    top = int(segment["r0"])
    left = int(segment["col"])
    span = int(segment["r1"] - segment["r0"] + 1)
    return (top, left, 1, span)


def _recover_segments(region: np.ndarray) -> list[dict]:
    segments: list[dict] = []
    for color in np.unique(region):
        color = int(color)
        if color == 0:
            continue
        coords = np.argwhere(region == color)
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])

        if len(rows) == 1:
            segments.append(
                {
                    "kind": "h",
                    "row": int(rows[0]),
                    "c0": int(cols.min()),
                    "c1": int(cols.max()),
                    "color": color,
                }
            )
            continue

        if len(cols) == 1:
            segments.append(
                {
                    "kind": "v",
                    "col": int(cols[0]),
                    "r0": int(rows.min()),
                    "r1": int(rows.max()),
                    "color": color,
                }
            )
            continue

        raise AssertionError("each color must form one axis-aligned line segment")

    segments.sort(key=_segment_sort_key)
    return segments


def _paint_segment(canvas: np.ndarray, segment: dict, col_offset: int = 0) -> None:
    color = int(segment["color"])
    if segment["kind"] == "h":
        r = int(segment["row"])
        c0 = int(segment["c0"]) + int(col_offset)
        c1 = int(segment["c1"]) + int(col_offset)
        canvas[r, c0 : c1 + 1] = color
    else:
        c = int(segment["col"]) + int(col_offset)
        r0 = int(segment["r0"])
        r1 = int(segment["r1"])
        canvas[r0 : r1 + 1, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose an overlaid set of colored horizontal/vertical line segments into separate panels.

    Input structure:
    - Black background.
    - A colored pattern is the overlay of several single-cell-thick horizontal
      and vertical line segments with unique colors.
    - Later segments overwrite earlier ones at crossings.

    Rule:
    - Recover each full line segment from its visible same-color collinear cells.
    - Sort segments from top to bottom, then left to right.
    - Draw one recovered segment per panel.
    - Panels use the cropped pattern size and are separated by grey columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    region = _crop_non_black(g)
    h, w = region.shape
    segments = _recover_segments(region)
    assert segments, "expected at least one segment"

    sep = 5
    out_w = len(segments) * w + max(0, len(segments) - 1)
    out = np.zeros((h, out_w), dtype=int)

    for idx, segment in enumerate(segments):
        panel_c0 = idx * (w + 1)
        if idx > 0:
            out[:, panel_c0 - 1] = sep
        _paint_segment(out, segment, col_offset=panel_c0)

    return out


# --------------- generator ---------------

_LINE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_BG = 0
_SEP = 5
_REGION_H_RANGE = (6, 10)
_REGION_W_RANGE = (6, 10)
_MARGIN_RANGE = (0, 3)
_TOTAL_LINES_RANGE = (3, 6)
_MAX_LINES_PER_AXIS = 3
_MAX_GENERATION_ATTEMPTS = 400


def _line_length(line: dict) -> int:
    if line["kind"] == "h":
        return int(line["c1"] - line["c0"] + 1)
    return int(line["r1"] - line["r0"] + 1)


def _line_cells(line: dict) -> list[tuple[int, int]]:
    if line["kind"] == "h":
        r = int(line["row"])
        return [(r, c) for c in range(int(line["c0"]), int(line["c1"]) + 1)]
    c = int(line["col"])
    return [(r, c) for r in range(int(line["r0"]), int(line["r1"]) + 1)]


def _crossing_point(a: dict, b: dict) -> tuple[int, int] | None:
    if a["kind"] == b["kind"]:
        return None
    h = a if a["kind"] == "h" else b
    v = b if a["kind"] == "h" else a
    r = int(h["row"])
    c = int(v["col"])
    if int(h["c0"]) <= c <= int(h["c1"]) and int(v["r0"]) <= r <= int(v["r1"]):
        return (r, c)
    return None


def _valid_structure(lines: list[dict]) -> bool:
    crossings_per_line = [0 for _ in lines]
    total_crossings = 0

    for i, a in enumerate(lines):
        for j in range(i + 1, len(lines)):
            b = lines[j]
            pt = _crossing_point(a, b)
            if pt is None:
                continue

            h = a if a["kind"] == "h" else b
            v = b if a["kind"] == "h" else a
            r, c = pt

            if c in (int(h["c0"]), int(h["c1"])) or r in (int(v["r0"]), int(v["r1"])):
                return False

            crossings_per_line[i] += 1
            crossings_per_line[j] += 1
            total_crossings += 1

    if total_crossings == 0:
        return False

    for line, crossing_count in zip(lines, crossings_per_line):
        if crossing_count > _line_length(line) - 2:
            return False

    return True


def _tight_bbox(lines: list[dict]) -> tuple[int, int, int, int]:
    pts = [cell for line in lines for cell in _line_cells(line)]
    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _shift_to_tight_region(lines: list[dict]) -> tuple[list[dict], tuple[int, int]]:
    r0, c0, r1, c1 = _tight_bbox(lines)
    shifted: list[dict] = []
    for line in lines:
        item = dict(line)
        if item["kind"] == "h":
            item["row"] = int(item["row"] - r0)
            item["c0"] = int(item["c0"] - c0)
            item["c1"] = int(item["c1"] - c0)
        else:
            item["col"] = int(item["col"] - c0)
            item["r0"] = int(item["r0"] - r0)
            item["r1"] = int(item["r1"] - r0)
        shifted.append(item)
    return shifted, (r1 - r0 + 1, c1 - c0 + 1)


def _render_overlay(lines: list[dict], shape: tuple[int, int], draw_order: list[int]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=int)
    for idx in draw_order:
        _paint_segment(canvas, lines[idx])
    return canvas


def _render_expected(lines: list[dict], shape: tuple[int, int]) -> np.ndarray:
    ordered = sorted(lines, key=_segment_sort_key)
    out = np.full((shape[0], len(ordered) * shape[1] + len(ordered) - 1), _BG, dtype=int)
    for idx, line in enumerate(ordered):
        panel_c0 = idx * (shape[1] + 1)
        if idx > 0:
            out[:, panel_c0 - 1] = _SEP
        _paint_segment(out, line, col_offset=panel_c0)
    return out


def _sample_counts(max_h: int, max_v: int) -> tuple[int, int] | None:
    low_total, high_total = _TOTAL_LINES_RANGE
    total = random.randint(low_total, min(high_total, max_h + max_v))
    lo_h = max(1, total - max_v)
    hi_h = min(max_h, total - 1)
    if lo_h > hi_h:
        return None
    num_h = random.randint(lo_h, hi_h)
    num_v = total - num_h
    return num_h, num_v


def _sample_lines(region_h: int, region_w: int) -> list[dict] | None:
    max_h = min(_MAX_LINES_PER_AXIS, region_h - 2)
    max_v = min(_MAX_LINES_PER_AXIS, region_w - 2)
    counts = _sample_counts(max_h, max_v)
    if counts is None:
        return None
    num_h, num_v = counts

    rows = sorted(random.sample(range(region_h), num_h))
    cols = sorted(random.sample(range(region_w), num_v))
    colors = random.sample(_LINE_COLORS, num_h + num_v)

    lines: list[dict] = []
    color_idx = 0

    for row in rows:
        length = random.randint(3, region_w)
        c0 = random.randint(0, region_w - length)
        lines.append(
            {
                "kind": "h",
                "row": row,
                "c0": c0,
                "c1": c0 + length - 1,
                "color": colors[color_idx],
            }
        )
        color_idx += 1

    for col in cols:
        length = random.randint(3, region_h)
        r0 = random.randint(0, region_h - length)
        lines.append(
            {
                "kind": "v",
                "col": col,
                "r0": r0,
                "r1": r0 + length - 1,
                "color": colors[color_idx],
            }
        )
        color_idx += 1

    if not _valid_structure(lines):
        return None
    return lines


def _embed_region(region: np.ndarray) -> np.ndarray:
    top = random.randint(*_MARGIN_RANGE)
    bottom = random.randint(*_MARGIN_RANGE)
    left = random.randint(*_MARGIN_RANGE)
    right = random.randint(*_MARGIN_RANGE)

    out = np.zeros((top + region.shape[0] + bottom, left + region.shape[1] + right), dtype=int)
    out[top : top + region.shape[0], left : left + region.shape[1]] = region
    return out


def _gen_attempt() -> np.ndarray | None:
    region_h = random.randint(*_REGION_H_RANGE)
    region_w = random.randint(*_REGION_W_RANGE)
    lines = _sample_lines(region_h, region_w)
    if lines is None:
        return None

    tight_lines, tight_shape = _shift_to_tight_region(lines)
    draw_order = list(range(len(tight_lines)))
    random.shuffle(draw_order)
    overlay = _render_overlay(tight_lines, tight_shape, draw_order)

    # Each line must remain recoverable from the visible overlay.
    for line in tight_lines:
        if np.count_nonzero(overlay == int(line["color"])) < 2:
            return None

    inp = _embed_region(overlay)
    expected = _render_expected(tight_lines, tight_shape)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid overlaid-line puzzle")
