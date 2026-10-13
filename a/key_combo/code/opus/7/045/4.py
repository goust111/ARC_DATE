from __future__ import annotations

from collections import Counter

import numpy as np


GREY = 5
MARKER_COLORS = (1, 3, 4)
GRID_MIN = 16
GRID_MAX_EXCL = 23
RECT_H_RANGE = range(3, 7)
RECT_W_RANGE = range(2, 5)
GEN_RETRY_LIMIT = 500

BBox = tuple[int, int, int, int]
Segment = dict[str, int | str]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _bbox_gap_ok(box: BBox, others: list[BBox], gap: int) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if (
            r0 <= or1 + gap
            and or0 <= r1 + gap
            and c0 <= oc1 + gap
            and oc0 <= c1 + gap
        ):
            return False
    return True


def _segment_touches_bbox(seg: Segment, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    if seg["orient"] == "h":
        return r0 <= int(seg["r"]) <= r1 and _intervals_overlap(int(seg["c0"]), int(seg["c1"]), c0, c1)
    return c0 <= int(seg["c"]) <= c1 and _intervals_overlap(int(seg["r0"]), int(seg["r1"]), r0, r1)


def _segment_cuts_bbox(seg: Segment, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    if seg["orient"] == "h":
        return r0 < int(seg["r"]) < r1 and _intervals_overlap(int(seg["c0"]), int(seg["c1"]), c0, c1)
    return c0 < int(seg["c"]) < c1 and _intervals_overlap(int(seg["r0"]), int(seg["r1"]), r0, r1)


def _output_envelope(box: BBox, mode: str) -> BBox:
    r0, c0, r1, c1 = box
    if mode == "h":
        return r0 - 1, c0, r1 + 1, c1
    if mode == "v":
        return r0, c0 - 1, r1, c1 + 1
    return box


def _component_bboxes(grid: np.ndarray, color: int) -> list[BBox]:
    mask = np.asarray(grid == color, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    h, w = map(int, grid.shape)
    boxes: list[BBox] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            box = (min(rows), min(cols), max(rows), max(cols))
            r0, c0, r1, c1 = box
            assert len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1), "grey objects must be solid rectangles"
            boxes.append(box)
    return sorted(boxes)


def _extract_segments(grid: np.ndarray, bg: int) -> list[Segment]:
    counts = Counter(map(int, grid.ravel()))
    segments: list[Segment] = []
    for color, count in sorted(counts.items()):
        if color in (bg, GREY):
            continue
        assert count == 2, "each marker color must occur exactly twice"
        coords = np.argwhere(grid == color)
        (r0, c0), (r1, c1) = coords.tolist()
        if r0 == r1:
            segments.append(
                {
                    "id": len(segments),
                    "orient": "h",
                    "color": int(color),
                    "r": int(r0),
                    "c0": int(min(c0, c1)),
                    "c1": int(max(c0, c1)),
                }
            )
        else:
            assert c0 == c1, "same-colored markers must share a row or a column"
            segments.append(
                {
                    "id": len(segments),
                    "orient": "v",
                    "color": int(color),
                    "c": int(c0),
                    "r0": int(min(r0, r1)),
                    "r1": int(max(r0, r1)),
                }
            )
    assert segments, "expected at least one marker pair"
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-colored marker pairs and split any crossed grey rectangle around the line path.

    Input structure:
    - The background is black or another dominant color.
    - Non-grey foreground markers come in same-colored pairs, with each pair aligned on one row or one column.
    - Grey foreground objects are solid axis-aligned rectangles, and each rectangle is crossed by at most one line.

    Rule:
    - Recover one horizontal or vertical line segment from each same-colored marker pair and keep the endpoints included.
    - Before drawing the lines, split every crossed grey rectangle: a horizontal line opens a one-row gap on its row, and a vertical line opens a one-column gap on its column, with the two rectangle halves shifted one cell outward.
    - After all rectangle splits, draw all horizontal lines and then all vertical lines so vertical colors win at crossings.

    Key insight:
    - The sparse colored pixels specify straight line paths, and those paths determine both where rectangles open and which color survives at line intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    segments = _extract_segments(g, bg)
    rectangles = _component_bboxes(g, GREY)

    out = g.copy()
    out[g == GREY] = bg

    for box in rectangles:
        hits = [seg for seg in segments if _segment_cuts_bbox(seg, box)]
        assert len(hits) <= 1, "each rectangle may be crossed by at most one line"
        if not hits:
            r0, c0, r1, c1 = box
            out[r0 : r1 + 1, c0 : c1 + 1] = GREY
            continue

        seg = hits[0]
        r0, c0, r1, c1 = box
        if seg["orient"] == "h":
            cut_r = int(seg["r"])
            assert r0 >= 1 and r1 + 1 < g.shape[0], "split rectangle must stay in bounds after moving"
            out[r0 - 1 : cut_r, c0 : c1 + 1] = GREY
            out[cut_r + 1 : r1 + 2, c0 : c1 + 1] = GREY
        else:
            cut_c = int(seg["c"])
            assert c0 >= 1 and c1 + 1 < g.shape[1], "split rectangle must stay in bounds after moving"
            out[r0 : r1 + 1, c0 - 1 : cut_c] = GREY
            out[r0 : r1 + 1, cut_c + 1 : c1 + 2] = GREY

    for seg in segments:
        if seg["orient"] == "h":
            out[int(seg["r"]), int(seg["c0"]) : int(seg["c1"]) + 1] = int(seg["color"])
    for seg in segments:
        if seg["orient"] == "v":
            out[int(seg["r0"]) : int(seg["r1"]) + 1, int(seg["c"])] = int(seg["color"])
    return out


def _sample_covering_span(limit: int, include: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for start in range(1, limit - 1):
        for end in range(start + 6, limit - 1):
            if start < include < end:
                candidates.append((start, end))
    if not candidates:
        raise ValueError("cannot place span")
    return _choose_one(candidates)


def _sample_free_span(limit: int) -> tuple[int, int]:
    candidates = [(start, end) for start in range(1, limit - 1) for end in range(start + 6, limit - 1)]
    if not candidates:
        raise ValueError("cannot place span")
    return _choose_one(candidates)


def _count_crossings(segments: list[Segment]) -> int:
    horizontals = [seg for seg in segments if seg["orient"] == "h"]
    verticals = [seg for seg in segments if seg["orient"] == "v"]
    total = 0
    for h_seg in horizontals:
        for v_seg in verticals:
            if int(h_seg["c0"]) <= int(v_seg["c"]) <= int(h_seg["c1"]) and int(v_seg["r0"]) <= int(h_seg["r"]) <= int(v_seg["r1"]):
                total += 1
    return total


def _sample_segments(height: int, width: int) -> list[Segment] | None:
    h_count = int(np.random.choice([1, 2]))
    v_count = 3 - h_count
    row_pool = np.arange(3, height - 3)
    col_pool = np.arange(3, width - 3)
    if len(row_pool) < h_count or len(col_pool) < v_count:
        return None

    h_rows = np.random.choice(row_pool, size=h_count, replace=False).astype(int).tolist()
    v_cols = np.random.choice(col_pool, size=v_count, replace=False).astype(int).tolist()
    colors = np.random.permutation(np.array(MARKER_COLORS, dtype=int)).tolist()

    segments: list[Segment] = []
    first_h_row = h_rows[0]
    first_v_col = v_cols[0]
    try:
        c0, c1 = _sample_covering_span(width, first_v_col)
    except ValueError:
        return None
    segments.append({"id": 0, "orient": "h", "color": colors[0], "r": first_h_row, "c0": c0, "c1": c1})

    for idx, r in enumerate(h_rows[1:], start=1):
        try:
            if np.random.rand() < 0.5:
                c0, c1 = _sample_covering_span(width, first_v_col)
            else:
                c0, c1 = _sample_free_span(width)
        except ValueError:
            return None
        segments.append({"id": idx, "orient": "h", "color": colors[idx], "r": int(r), "c0": c0, "c1": c1})

    base = h_count
    try:
        r0, r1 = _sample_covering_span(height, first_h_row)
    except ValueError:
        return None
    segments.append({"id": base, "orient": "v", "color": colors[base], "c": first_v_col, "r0": r0, "r1": r1})

    for idx, c in enumerate(v_cols[1:], start=1):
        seg_id = base + idx
        try:
            if np.random.rand() < 0.5:
                r0, r1 = _sample_covering_span(height, first_h_row)
            else:
                r0, r1 = _sample_free_span(height)
        except ValueError:
            return None
        segments.append({"id": seg_id, "orient": "v", "color": colors[seg_id], "c": int(c), "r0": r0, "r1": r1})

    crossings = _count_crossings(segments)
    if crossings < 1 or crossings > 2:
        return None
    return segments


def _cut_candidates(
    height: int,
    width: int,
    mode: str,
    seg: Segment,
    segments: list[Segment],
    input_boxes: list[BBox],
    output_boxes: list[BBox],
) -> list[tuple[BBox, BBox]]:
    candidates: list[tuple[BBox, BBox]] = []
    for rect_h in RECT_H_RANGE:
        for rect_w in RECT_W_RANGE:
            if mode == "v" and rect_w < 3:
                continue
            for r0 in range(height - rect_h + 1):
                r1 = r0 + rect_h - 1
                if mode == "h" and (r0 < 1 or r1 > height - 2):
                    continue
                for c0 in range(width - rect_w + 1):
                    c1 = c0 + rect_w - 1
                    if mode == "v" and (c0 < 1 or c1 > width - 2):
                        continue
                    box = (r0, c0, r1, c1)
                    env = _output_envelope(box, mode)
                    if not _bbox_gap_ok(box, input_boxes, gap=1):
                        continue
                    if not _bbox_gap_ok(env, output_boxes, gap=0):
                        continue

                    if mode == "h":
                        if seg["orient"] != "h":
                            continue
                        if not (r0 < int(seg["r"]) < r1 and int(seg["c0"]) < c0 and c1 < int(seg["c1"])):
                            continue
                    else:
                        if seg["orient"] != "v":
                            continue
                        if not (c0 < int(seg["c"]) < c1 and int(seg["r0"]) < r0 and r1 < int(seg["r1"])):
                            continue

                    bad = False
                    for other in segments:
                        if int(other["id"]) == int(seg["id"]):
                            continue
                        if _segment_touches_bbox(other, env):
                            bad = True
                            break
                    if not bad:
                        candidates.append((box, env))
    return candidates


def _uncut_candidates(
    height: int,
    width: int,
    segments: list[Segment],
    input_boxes: list[BBox],
    output_boxes: list[BBox],
) -> list[tuple[BBox, BBox]]:
    candidates: list[tuple[BBox, BBox]] = []
    for rect_h in RECT_H_RANGE:
        for rect_w in RECT_W_RANGE:
            for r0 in range(height - rect_h + 1):
                r1 = r0 + rect_h - 1
                for c0 in range(width - rect_w + 1):
                    c1 = c0 + rect_w - 1
                    box = (r0, c0, r1, c1)
                    if not _bbox_gap_ok(box, input_boxes, gap=1):
                        continue
                    if not _bbox_gap_ok(box, output_boxes, gap=0):
                        continue
                    if any(_segment_touches_bbox(seg, box) for seg in segments):
                        continue
                    candidates.append((box, box))
    return candidates


def _choose_one(items: list):
    return items[int(np.random.randint(0, len(items)))]


def _render_input(height: int, width: int, segments: list[Segment], rectangles: list[BBox]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for r0, c0, r1, c1 in rectangles:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    for seg in segments:
        color = int(seg["color"])
        if seg["orient"] == "h":
            grid[int(seg["r"]), int(seg["c0"])] = color
            grid[int(seg["r"]), int(seg["c1"])] = color
        else:
            grid[int(seg["r0"]), int(seg["c"])] = color
            grid[int(seg["r1"]), int(seg["c"])] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(GRID_MIN, GRID_MAX_EXCL))
    width = int(np.random.randint(GRID_MIN, GRID_MAX_EXCL))
    segments = _sample_segments(height, width)
    if segments is None:
        return None

    horizontals = [seg for seg in segments if seg["orient"] == "h"]
    verticals = [seg for seg in segments if seg["orient"] == "v"]
    pattern = _choose_one([("h",), ("v",), ("h", "v")])
    if "h" in pattern and not horizontals:
        return None
    if "v" in pattern and not verticals:
        return None

    rect_count = int(np.random.randint(max(2, len(pattern)), 4))
    extra_count = rect_count - len(pattern)
    input_boxes: list[BBox] = []
    output_boxes: list[BBox] = []
    rectangles: list[BBox] = []
    used_seg_ids: set[int] = set()

    for mode in pattern:
        pool = horizontals if mode == "h" else verticals
        pool = [seg for seg in pool if int(seg["id"]) not in used_seg_ids]
        if not pool:
            return None
        np.random.shuffle(pool)
        placed = False
        for seg in pool:
            candidates = _cut_candidates(height, width, mode, seg, segments, input_boxes, output_boxes)
            if not candidates:
                continue
            box, env = _choose_one(candidates)
            rectangles.append(box)
            input_boxes.append(box)
            output_boxes.append(env)
            used_seg_ids.add(int(seg["id"]))
            placed = True
            break
        if not placed:
            return None

    for _ in range(extra_count):
        candidates = _uncut_candidates(height, width, segments, input_boxes, output_boxes)
        if not candidates:
            return None
        box, env = _choose_one(candidates)
        rectangles.append(box)
        input_boxes.append(box)
        output_boxes.append(env)

    grid = _render_input(height, width, segments, rectangles)

    counts = Counter(map(int, grid.ravel()))
    if any(counts.get(color, 0) != 2 for color in MARKER_COLORS):
        return None
    if counts.get(GREY, 0) == 0:
        return None

    boxes = _component_bboxes(grid, GREY)
    if len(boxes) != rect_count:
        return None
    hits_per_box = [sum(_segment_cuts_bbox(seg, box) for seg in segments) for box in boxes]
    if any(hit > 1 for hit in hits_per_box) or max(hits_per_box, default=0) < 1:
        return None
    if not horizontals or not verticals or _count_crossings(segments) < 1:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-4 input")
