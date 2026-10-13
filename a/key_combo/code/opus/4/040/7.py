from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
LINE_COLORS = (BLUE, RED)

BBox = tuple[int, int, int, int]


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _is_rectangle_ring(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    actual = {tuple(map(int, rc)) for rc in coords}
    return actual == expected


def _find_rectangles(grid: np.ndarray) -> list[BBox]:
    rects = []
    for coords in _component_coords(grid == GRAY):
        assert _is_rectangle_ring(coords), "expected every gray component to be a 1-cell-thick rectangle frame"
        rects.append(_bbox(coords))
    rects.sort()
    assert rects, "expected at least one gray rectangle"
    return rects


def _detect_lines(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    lines: list[dict] = []
    for r in range(1, h - 1):
        color = int(grid[r, 0])
        if color in LINE_COLORS and int(grid[r, w - 1]) == color:
            lines.append({"kind": "h", "pos": r, "color": color})
    for c in range(1, w - 1):
        color = int(grid[0, c])
        if color in LINE_COLORS and int(grid[h - 1, c]) == color:
            lines.append({"kind": "v", "pos": c, "color": color})
    assert 2 <= len(lines) <= 3, "expected 2 or 3 boundary marker pairs"
    return lines


def _paint_lines(base: np.ndarray, lines: list[dict]) -> np.ndarray:
    out = base.copy()
    for line in lines:
        color = int(line["color"])
        if line["kind"] == "h":
            r = int(line["pos"])
            mask = base[r, :] == BLACK
            out[r, mask] = color
        else:
            c = int(line["pos"])
            mask = base[:, c] == BLACK
            out[mask, c] = color
    return out


def _contact_records(line_grid: np.ndarray, box: BBox) -> list[tuple[int, int, int, int, str, int]]:
    r0, c0, r1, c1 = box
    records: list[tuple[int, int, int, int, str, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if not (r in (r0, r1) or c in (c0, c1)):
                continue
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(line_grid.shape, nr, nc):
                    continue
                color = int(line_grid[nr, nc])
                if color not in LINE_COLORS:
                    continue
                if nr == r - 1:
                    side = "top"
                elif nr == r + 1:
                    side = "bottom"
                elif nc == c - 1:
                    side = "left"
                else:
                    side = "right"
                records.append((r, c, nr, nc, side, color))
    return records


def _fill_box(out: np.ndarray, box: BBox, side: str, line_color: int) -> None:
    r0, c0, r1, c1 = box
    seq = (RED, YELLOW) if line_color == RED else (BLUE, GREEN)
    if side in {"top", "bottom"}:
        for idx, c in enumerate(range(c0 + 1, c1)):
            fill_color = seq[idx % 2]
            for r in range(r0 + 1, r1):
                if out[r, c] == BLACK:
                    out[r, c] = fill_color
    else:
        for idx, r in enumerate(range(r0 + 1, r1)):
            fill_color = seq[idx % 2]
            for c in range(c0 + 1, c1):
                if out[r, c] == BLACK:
                    out[r, c] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw boundary-guided lines, select the gray rectangle frames they touch, and fill each selected interior by a side-coded scan direction and line-coded color cycle.

    Input structure:
    - The grid has a black background with several non-touching gray hollow rectangles of thickness 1.
    - Matching boundary markers on left/right or top/bottom edges define horizontal or vertical guide lines.
    - Marker colors are blue or red, and rectangle interiors start black.

    Rule:
    - Extend each boundary marker pair into a full-row or full-column guide line of the same color, painting only black cells and never overwriting gray frame pixels.
    - A gray frame is selected when any of its border pixels is 4-neighbor-adjacent to a guide-line pixel.
    - Use the earliest contact point in reading order to choose the controlling line: contacts on the top or bottom side mean fill the interior column-by-column from left to right; contacts on the left or right side mean fill row-by-row from top to bottom.
    - A red controlling line uses the alternating sequence [red, yellow], while a blue controlling line uses [blue, green]; guide-line pixels inside boxes stay unchanged.

    Key insight:
    - The same guide line simultaneously encodes selection, scan direction, and the alternating fill palette, with the deciding side coming from the earliest border contact.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    rects = _find_rectangles(g)
    lines = _detect_lines(g)
    out = _paint_lines(g, lines)

    for box in rects:
        records = _contact_records(out, box)
        if not records:
            continue
        r, c, nr, nc, side, color = min(records, key=lambda item: item[:4])
        _ = (r, c, nr, nc)
        _fill_box(out, box, side, color)
    return out


GEN_MIN_SIZE = 24
GEN_MAX_SIZE = 29
GEN_RECT_COUNTS = (4, 5)
GEN_MIN_SIDE = 5
GEN_MAX_SIDE_FOR_4 = 8
GEN_MAX_SIDE_FOR_5 = 7
GEN_ATTEMPTS = 400
GEN_LINE_ATTEMPTS = 120


def _boxes_separate(a: BBox, b: BBox, pad: int = 1) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _sample_boxes(h: int, w: int, n_rects: int) -> list[BBox] | None:
    max_side = GEN_MAX_SIDE_FOR_5 if n_rects == 5 else GEN_MAX_SIDE_FOR_4
    dims = [
        (random.randint(GEN_MIN_SIDE, max_side), random.randint(GEN_MIN_SIDE, max_side))
        for _ in range(n_rects)
    ]
    dims.sort(key=lambda hw: hw[0] * hw[1], reverse=True)
    boxes: list[BBox] = []
    for box_h, box_w in dims:
        placed = False
        for _ in range(250):
            r0 = random.randint(2, h - box_h - 3)
            c0 = random.randint(2, w - box_w - 3)
            box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if all(_boxes_separate(box, other, pad=1) for other in boxes):
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None
    boxes.sort()
    return boxes


def _render_input(h: int, w: int, boxes: list[BBox], lines: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r0, c0, r1, c1 in boxes:
        grid[r0, c0 : c1 + 1] = GRAY
        grid[r1, c0 : c1 + 1] = GRAY
        grid[r0 : r1 + 1, c0] = GRAY
        grid[r0 : r1 + 1, c1] = GRAY
    for line in lines:
        color = int(line["color"])
        if line["kind"] == "h":
            r = int(line["pos"])
            grid[r, 0] = color
            grid[r, w - 1] = color
        else:
            c = int(line["pos"])
            grid[0, c] = color
            grid[h - 1, c] = color
    return grid


def _safe_inner_rows(box: BBox) -> list[int]:
    r0, _c0, r1, _c1 = box
    return list(range(r0 + 2, r1 - 1))


def _safe_inner_cols(box: BBox) -> list[int]:
    _r0, c0, _r1, c1 = box
    return list(range(c0 + 2, c1 - 1))


def _outer_rows(box: BBox, h: int) -> list[int]:
    r0, _c0, r1, _c1 = box
    rows = []
    if 1 <= r0 - 1 < h - 1:
        rows.append(r0 - 1)
    if 1 <= r1 + 1 < h - 1:
        rows.append(r1 + 1)
    return rows


def _outer_cols(box: BBox, w: int) -> list[int]:
    _r0, c0, _r1, c1 = box
    cols = []
    if 1 <= c0 - 1 < w - 1:
        cols.append(c0 - 1)
    if 1 <= c1 + 1 < w - 1:
        cols.append(c1 + 1)
    return cols


def _all_extra_candidates(boxes: list[BBox], h: int, w: int, color: int, used: set[tuple[str, int]]) -> list[dict]:
    specs: set[tuple[str, int]] = set()
    for box in boxes:
        for r in _safe_inner_rows(box):
            specs.add(("h", r))
        for r in _outer_rows(box, h):
            specs.add(("h", r))
        for c in _safe_inner_cols(box):
            specs.add(("v", c))
        for c in _outer_cols(box, w):
            specs.add(("v", c))
    return [{"kind": kind, "pos": pos, "color": color} for kind, pos in sorted(specs) if (kind, pos) not in used]


def _line_pixels(base: np.ndarray, line: dict) -> set[tuple[int, int]]:
    h, w = base.shape
    if line["kind"] == "h":
        r = int(line["pos"])
        return {(r, c) for c in range(w) if int(base[r, c]) != GRAY}
    c = int(line["pos"])
    return {(r, c) for r in range(h) if int(base[r, c]) != GRAY}


def _line_box_contacts(base: np.ndarray, box: BBox, line: dict) -> list[dict]:
    pixels = _line_pixels(base, line)
    r0, c0, r1, c1 = box
    contacts: list[dict] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if not (r in (r0, r1) or c in (c0, c1)):
                continue
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in pixels:
                    continue
                if nr == r - 1:
                    side = "top"
                elif nr == r + 1:
                    side = "bottom"
                elif nc == c - 1:
                    side = "left"
                else:
                    side = "right"
                contacts.append(
                    {
                        "point": (r, c),
                        "neighbor": (nr, nc),
                        "side": side,
                        "kind": str(line["kind"]),
                        "color": int(line["color"]),
                    }
                )
    return contacts


def _analyze_layout(base: np.ndarray, boxes: list[BBox], lines: list[dict]) -> dict:
    per_line_hits = [set() for _ in lines]
    per_box_contacts: list[list[dict]] = [[] for _ in boxes]
    for line_idx, line in enumerate(lines):
        for box_idx, box in enumerate(boxes):
            contacts = _line_box_contacts(base, box, line)
            if contacts:
                per_line_hits[line_idx].add(box_idx)
                per_box_contacts[box_idx].extend(contacts)

    chosen: list[dict | None] = []
    for contacts in per_box_contacts:
        if not contacts:
            chosen.append(None)
            continue
        ranked = sorted(contacts, key=lambda item: item["point"] + item["neighbor"])
        if len(ranked) >= 2 and ranked[0]["point"] == ranked[1]["point"] and ranked[0]["neighbor"] != ranked[1]["neighbor"]:
            return {"valid": False}
        chosen.append(ranked[0])

    return {
        "valid": True,
        "per_line_hits": per_line_hits,
        "per_box_contacts": per_box_contacts,
        "chosen": chosen,
    }


def _render_expected(base: np.ndarray, boxes: list[BBox], lines: list[dict], analysis: dict) -> np.ndarray:
    out = _paint_lines(base, lines)
    for box, choice in zip(boxes, analysis["chosen"]):
        if choice is None:
            continue
        _fill_box(out, box, str(choice["side"]), int(choice["color"]))
    return out


def _validate_layout(base: np.ndarray, boxes: list[BBox], lines: list[dict], need_multi: bool) -> bool:
    if not (4 <= len(boxes) <= 5 and 2 <= len(lines) <= 3):
        return False
    if not any(line["kind"] == "h" for line in lines):
        return False
    if not any(line["kind"] == "v" for line in lines):
        return False

    analysis = _analyze_layout(base, boxes, lines)
    if not analysis["valid"]:
        return False
    if any(not hits for hits in analysis["per_line_hits"]):
        return False

    touched_by_h = set()
    touched_by_v = set()
    for line, hits in zip(lines, analysis["per_line_hits"]):
        if line["kind"] == "h":
            touched_by_h |= hits
        else:
            touched_by_v |= hits
    if not touched_by_h or not touched_by_v:
        return False

    chosen = analysis["chosen"]
    chosen_h = sum(choice is not None and choice["kind"] == "h" for choice in chosen)
    chosen_v = sum(choice is not None and choice["kind"] == "v" for choice in chosen)
    untouched = sum(choice is None for choice in chosen)
    multi = sum(len(contacts) >= 2 for contacts in analysis["per_box_contacts"])
    if chosen_h == 0 or chosen_v == 0:
        return False
    if untouched == 0:
        return False
    if need_multi and multi == 0:
        return False

    expected = _render_expected(base, boxes, lines, analysis)
    if np.array_equal(base, expected):
        return False
    if np.count_nonzero(expected != base) < 18:
        return False

    actual = main(base)
    return np.array_equal(actual, expected)


def _sample_lines(boxes: list[BBox], h: int, w: int) -> list[dict] | None:
    color = random.choice(LINE_COLORS)
    h_box = random.choice(boxes)
    v_box = random.choice(boxes)
    h_rows = _safe_inner_rows(h_box)
    v_cols = _safe_inner_cols(v_box)
    if not h_rows or not v_cols:
        return None

    lines = [
        {"kind": "h", "pos": random.choice(h_rows), "color": color},
        {"kind": "v", "pos": random.choice(v_cols), "color": color},
    ]
    if random.random() < 0.65:
        used = {(str(line["kind"]), int(line["pos"])) for line in lines}
        extra = _all_extra_candidates(boxes, h, w, color, used)
        if extra:
            lines.append(random.choice(extra))
    return lines


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(GEN_MIN_SIZE, GEN_MAX_SIZE)
        w = random.randint(GEN_MIN_SIZE, GEN_MAX_SIZE)
        n_rects = random.choice(GEN_RECT_COUNTS)
        boxes = _sample_boxes(h, w, n_rects)
        if boxes is None:
            continue

        need_multi = random.random() < 0.5
        for _ in range(GEN_LINE_ATTEMPTS):
            lines = _sample_lines(boxes, h, w)
            if lines is None:
                continue
            base = _render_input(h, w, boxes, lines)
            if _validate_layout(base, boxes, lines, need_multi):
                return base
    raise RuntimeError("failed to generate a valid puzzle input")
