from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7


PALETTE = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK)
GEN_ATTEMPTS = 600


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            part: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                part.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(part)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _box_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    row_gap = max(0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    col_gap = max(0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return row_gap, col_gap


def _indicator_of(rect: dict) -> tuple[int, int]:
    r0, c0, h, w, direction = rect["r0"], rect["c0"], rect["h"], rect["w"], rect["dir"]
    if direction == "left":
        return r0 + 1, c0
    if direction == "right":
        return r0 + 1, c0 + w - 1
    if direction == "up":
        return r0, c0 + 1
    return r0 + h - 1, c0 + 1


def _line_from_object(obj: dict, shape: tuple[int, int]) -> dict:
    h, w = shape
    r0, c0, r1, c1 = obj["bbox"]
    ir, ic = obj["indicator"]
    direction = obj["dir"]
    if direction == "left":
        return {
            "axis": "h",
            "dir": direction,
            "color": obj["color"],
            "freq": obj["freq"],
            "fixed": ir,
            "start": c0 - 1,
            "end": 0,
            "anchor": (ir, 0),
        }
    if direction == "right":
        return {
            "axis": "h",
            "dir": direction,
            "color": obj["color"],
            "freq": obj["freq"],
            "fixed": ir,
            "start": c1 + 1,
            "end": w - 1,
            "anchor": (ir, c1 + 1),
        }
    if direction == "up":
        return {
            "axis": "v",
            "dir": direction,
            "color": obj["color"],
            "freq": obj["freq"],
            "fixed": ic,
            "start": r0 - 1,
            "end": 0,
            "anchor": (0, ic),
        }
    return {
        "axis": "v",
        "dir": direction,
        "color": obj["color"],
        "freq": obj["freq"],
        "fixed": ic,
        "start": r1 + 1,
        "end": h - 1,
        "anchor": (r1 + 1, ic),
    }


def _iter_line(line: dict):
    if line["axis"] == "h":
        step = -1 if line["dir"] == "left" else 1
        dist = 1
        c = line["start"]
        while True:
            yield line["fixed"], c, dist
            if c == line["end"]:
                break
            c += step
            dist += 1
    else:
        step = -1 if line["dir"] == "up" else 1
        dist = 1
        r = line["start"]
        while True:
            yield r, line["fixed"], dist
            if r == line["end"]:
                break
            r += step
            dist += 1


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    counts = Counter(int(v) for v in g.ravel() if int(v) not in (Color.BLACK, Color.ORANGE))
    objects: list[dict] = []
    for cells in _components(g != Color.BLACK):
        r0, c0, r1, c1 = _bbox(cells)
        cell_set = set(cells)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                assert (rr, cc) in cell_set, "each component must occupy a full rectangle bbox"

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        orange = np.argwhere(sub == Color.ORANGE)
        assert len(orange) == 1, "expected one indicator pixel per rectangle"
        lr, lc = map(int, orange[0])
        ir, ic = r0 + lr, c0 + lc

        body_colors = {int(v) for v in sub.ravel() if int(v) not in (Color.BLACK, Color.ORANGE)}
        assert len(body_colors) == 1, "each rectangle must have one body color"
        color = next(iter(body_colors))

        candidates = []
        if ir == r0 and c0 < ic < c1:
            candidates.append("up")
        if ir == r1 and c0 < ic < c1:
            candidates.append("down")
        if ic == c0 and r0 < ir < r1:
            candidates.append("left")
        if ic == c1 and r0 < ir < r1:
            candidates.append("right")
        assert len(candidates) == 1, "indicator must lie on exactly one non-corner edge"

        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "indicator": (ir, ic),
                "dir": candidates[0],
                "color": color,
                "freq": counts[color],
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend indicator-marked rectangles into dashed rays and fill only cross-gaps with the rarer ray color.

    Input structure:
    - The grid is a black canvas containing 3-5 solid small rectangles in distinct colors.
    - Each rectangle has exactly one orange border pixel that marks the side of outward growth.
    - The orange pixel is preserved, and each rectangle's own color frequency is its body area in the input.

    Rule:
    - Find each rectangle and read the indicator side to determine one of four cardinal directions.
    - From the cell just outside that side, draw a one-pixel-wide dashed ray in the rectangle color to the canvas edge using color, black, color, black, ...
    - After all rays are drawn, fill a black cell only when it is simultaneously a horizontal dash gap and a vertical dash gap; choose the color whose source rectangle is globally rarer, breaking equal frequencies by the visually higher dashed ray.

    Key insight:
    - Ordinary dash gaps stay black; only perpendicular gap-gap crossings are filled, and the fill color depends on comparing whole-rectangle frequencies rather than local neighbors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    objects = _analyze_objects(g)
    lines = [_line_from_object(obj, g.shape) for obj in objects]
    by_color = {line["color"]: line for line in lines}

    for line in lines:
        for r, c, dist in _iter_line(line):
            if dist % 2 == 0:
                continue
            current = int(out[r, c])
            assert current in (Color.BLACK, line["color"]), "dash cells must not conflict"
            out[r, c] = line["color"]

    h, w = out.shape
    filled = out.copy()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(out[r, c]) != Color.BLACK:
                continue
            left = int(out[r, c - 1])
            right = int(out[r, c + 1])
            up = int(out[r - 1, c])
            down = int(out[r + 1, c])

            h_color = left if left == right and left not in (Color.BLACK, Color.ORANGE) else None
            v_color = up if up == down and up not in (Color.BLACK, Color.ORANGE) else None
            if h_color is None or v_color is None or h_color == v_color:
                continue

            h_line = by_color[h_color]
            v_line = by_color[v_color]
            if h_line["freq"] < v_line["freq"]:
                filled[r, c] = h_color
            elif v_line["freq"] < h_line["freq"]:
                filled[r, c] = v_color
            else:
                filled[r, c] = h_color if h_line["anchor"] <= v_line["anchor"] else v_color
    return filled


def _draw_input(rects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), Color.BLACK, dtype=int)
    for rect in rects:
        r0, c0, hh, ww, color = rect["r0"], rect["c0"], rect["h"], rect["w"], rect["color"]
        grid[r0 : r0 + hh, c0 : c0 + ww] = color
        ir, ic = _indicator_of(rect)
        grid[ir, ic] = Color.ORANGE
    return grid


def _shape_for_direction(direction: str, area_count: int) -> tuple[int, int]:
    if direction in ("left", "right"):
        return (3, 2) if area_count == 5 else (3, 3)
    return (2, 3) if area_count == 5 else (3, 3)


def _rand_even(low: int, high: int) -> int | None:
    values = [v for v in range(low, high + 1) if v % 2 == 0]
    if not values:
        return None
    return random.choice(values)


def _place_primary_pair(height: int, width: int, colors: list[int], tie_mode: bool) -> tuple[list[dict], tuple[int, int], int]:
    cross_r = random.randint(7, height - 8)
    cross_c = random.randint(7, width - 8)

    h_dir = random.choice(["left", "right"])
    v_dir = random.choice(["up", "down"])

    h_count = random.choice([5, 8])
    v_count = h_count if tie_mode else (8 if h_count == 5 else 5)
    h_h, h_w = _shape_for_direction(h_dir, h_count)
    v_h, v_w = _shape_for_direction(v_dir, v_count)

    max_h_offset = cross_c - h_w - 2 if h_dir == "right" else width - cross_c - h_w - 3
    max_v_offset = cross_r - v_h - 2 if v_dir == "down" else height - cross_r - v_h - 3
    h_offset = _rand_even(4, max_h_offset)
    v_offset = _rand_even(4, max_v_offset)
    if h_offset is None or v_offset is None:
        raise ValueError("primary pair has no feasible offset")

    if h_dir == "right":
        h_c1 = cross_c - h_offset
        h_c0 = h_c1 - h_w + 1
    else:
        h_c0 = cross_c + h_offset
    if h_dir == "left":
        h_c1 = h_c0 + h_w - 1
    h_r0 = cross_r - 1

    if v_dir == "down":
        v_r1 = cross_r - v_offset
        v_r0 = v_r1 - v_h + 1
    else:
        v_r0 = cross_r + v_offset
        v_r1 = v_r0 + v_h - 1
    v_c0 = cross_c - 1

    rect_h = {"r0": h_r0, "c0": h_c0, "h": h_h, "w": h_w, "dir": h_dir, "color": colors[0]}
    rect_v = {"r0": v_r0, "c0": v_c0, "h": v_h, "w": v_w, "dir": v_dir, "color": colors[1]}

    if tie_mode:
        line_h = _line_from_object(
            {
                "bbox": (rect_h["r0"], rect_h["c0"], rect_h["r0"] + rect_h["h"] - 1, rect_h["c0"] + rect_h["w"] - 1),
                "indicator": _indicator_of(rect_h),
                "dir": rect_h["dir"],
                "color": rect_h["color"],
                "freq": h_count,
            },
            (height, width),
        )
        line_v = _line_from_object(
            {
                "bbox": (rect_v["r0"], rect_v["c0"], rect_v["r0"] + rect_v["h"] - 1, rect_v["c0"] + rect_v["w"] - 1),
                "indicator": _indicator_of(rect_v),
                "dir": rect_v["dir"],
                "color": rect_v["color"],
                "freq": v_count,
            },
            (height, width),
        )
        expected = rect_h["color"] if line_h["anchor"] <= line_v["anchor"] else rect_v["color"]
    else:
        expected = rect_h["color"] if h_count < v_count else rect_v["color"]

    return [rect_h, rect_v], (cross_r, cross_c), expected


def _random_rect(height: int, width: int, color: int) -> dict:
    direction = random.choice(["left", "right", "up", "down"])
    count = random.choice([5, 8])
    rect_h, rect_w = _shape_for_direction(direction, count)
    if direction == "left":
        r0 = random.randint(1, height - rect_h - 2)
        c0 = random.randint(max(4, width // 2), width - rect_w - 2)
    elif direction == "right":
        r0 = random.randint(1, height - rect_h - 2)
        c0 = random.randint(1, max(1, width // 2 - rect_w - 1))
    elif direction == "up":
        r0 = random.randint(max(4, height // 2), height - rect_h - 2)
        c0 = random.randint(1, width - rect_w - 2)
    else:
        r0 = random.randint(1, max(1, height // 2 - rect_h - 1))
        c0 = random.randint(1, width - rect_w - 2)
    return {"r0": r0, "c0": c0, "h": rect_h, "w": rect_w, "dir": direction, "color": color}


def _valid_rect_layout(rects: list[dict], shape: tuple[int, int]) -> bool:
    h, w = shape
    boxes = []
    line_rows = set()
    line_cols = set()

    for rect in rects:
        r0, c0, hh, ww = rect["r0"], rect["c0"], rect["h"], rect["w"]
        r1, c1 = r0 + hh - 1, c0 + ww - 1
        if not (1 <= r0 and r1 <= h - 2 and 1 <= c0 and c1 <= w - 2):
            return False
        boxes.append((r0, c0, r1, c1))

        ir, ic = _indicator_of(rect)
        if rect["dir"] in ("left", "right"):
            if ir in line_rows:
                return False
            line_rows.add(ir)
            line_len = c0 if rect["dir"] == "left" else w - (c1 + 1)
        else:
            if ic in line_cols:
                return False
            line_cols.add(ic)
            line_len = r0 if rect["dir"] == "up" else h - (r1 + 1)
        if line_len < 4:
            return False

    for i, box_a in enumerate(boxes):
        for box_b in boxes[i + 1 :]:
            row_gap, col_gap = _box_gap(box_a, box_b)
            if row_gap < 2 or col_gap < 2:
                return False

    objects = []
    for rect, box in zip(rects, boxes):
        count = rect["h"] * rect["w"] - 1
        objects.append(
            {
                "bbox": box,
                "indicator": _indicator_of(rect),
                "dir": rect["dir"],
                "color": rect["color"],
                "freq": count,
            }
        )

    h_lines = [_line_from_object(obj, shape) for obj in objects if obj["dir"] in ("left", "right")]
    v_lines = [_line_from_object(obj, shape) for obj in objects if obj["dir"] in ("up", "down")]
    for h_line in h_lines:
        h_lo, h_hi = sorted((h_line["start"], h_line["end"]))
        for v_line in v_lines:
            v_lo, v_hi = sorted((v_line["start"], v_line["end"]))
            if not (h_lo <= v_line["fixed"] <= h_hi and v_lo <= h_line["fixed"] <= v_hi):
                continue
            h_dist = abs(v_line["fixed"] - (h_line["start"] - (-1 if h_line["dir"] == "left" else 1)))
            v_dist = abs(h_line["fixed"] - (v_line["start"] - (-1 if v_line["dir"] == "up" else 1)))
            if h_dist % 2 == 1 and v_dist % 2 == 1:
                return False
    return True


def _has_ordinary_gap(output: np.ndarray) -> bool:
    h, w = output.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(output[r, c]) != Color.BLACK:
                continue
            left = int(output[r, c - 1])
            right = int(output[r, c + 1])
            up = int(output[r - 1, c])
            down = int(output[r + 1, c])
            h_gap = left == right and left not in (Color.BLACK, Color.ORANGE)
            v_gap = up == down and up not in (Color.BLACK, Color.ORANGE)
            if h_gap != v_gap:
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    count = random.randint(3, 5)
    colors = random.sample(PALETTE, count)
    tie_mode = random.choice([False, True])

    try:
        rects, primary_cross, expected_color = _place_primary_pair(height, width, colors, tie_mode)
    except ValueError:
        return None

    for color in colors[2:]:
        placed = False
        for _ in range(80):
            candidate = _random_rect(height, width, color)
            trial = rects + [candidate]
            if _valid_rect_layout(trial, (height, width)):
                rects.append(candidate)
                placed = True
                break
        if not placed:
            return None

    if not _valid_rect_layout(rects, (height, width)):
        return None

    grid = _draw_input(rects, (height, width))
    if len(_components(grid != Color.BLACK)) != count:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    cr, cc = primary_cross
    if int(grid[cr, cc]) != Color.BLACK or int(out[cr, cc]) != expected_color:
        return None
    if not _has_ordinary_gap(out):
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid puzzle instance")
