from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400
_GENERATE_CALL_COUNT = 0
_BRANCH_CYCLE = ("vertical", "horizontal", "both")


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == bg:
                continue

            color = int(g[r0, c0])
            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while queue:
                r, c = queue.popleft()
                coords.append((int(r), int(c)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        queue.append((rr, cc))

            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            row_spans: dict[int, tuple[int, int]] = {}
            col_spans: dict[int, tuple[int, int]] = {}
            for r, c in coords:
                r_span = row_spans.get(r)
                c_span = col_spans.get(c)
                row_spans[r] = (c, c) if r_span is None else (min(r_span[0], c), max(r_span[1], c))
                col_spans[c] = (r, r) if c_span is None else (min(c_span[0], r), max(c_span[1], r))

            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "row_spans": row_spans,
                    "col_spans": col_spans,
                }
            )

    objects.sort(key=lambda obj: tuple(int(v) for v in obj["bbox"]))
    return objects


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> tuple[int, int] | None:
    lo = max(int(a0), int(b0))
    hi = min(int(a1), int(b1))
    if lo > hi:
        return None
    return int(lo), int(hi)


def _relation_type(a: dict, b: dict) -> str | None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    col_overlap = _interval_overlap(ac0, ac1, bc0, bc1)
    row_overlap = _interval_overlap(ar0, ar1, br0, br1)
    if col_overlap is not None and row_overlap is not None:
        return "both"
    if col_overlap is not None:
        return "vertical"
    if row_overlap is not None:
        return "horizontal"
    return None


def _midpoint(lo: int, hi: int) -> int:
    return (int(lo) + int(hi)) // 2


def _smaller_color(a: dict, b: dict) -> int:
    if int(a["area"]) < int(b["area"]):
        return int(a["color"])
    if int(b["area"]) < int(a["area"]):
        return int(b["color"])
    return int(a["color"]) if tuple(a["bbox"]) <= tuple(b["bbox"]) else int(b["color"])


def _vertical_gap(a: dict, b: dict, col: int) -> tuple[int, int] | None:
    a_box = tuple(int(v) for v in a["bbox"])
    b_box = tuple(int(v) for v in b["bbox"])
    a_center = a_box[0] + a_box[2]
    b_center = b_box[0] + b_box[2]
    if (a_center, a_box[0], a_box[1]) <= (b_center, b_box[0], b_box[1]):
        upper, lower = a, b
    else:
        upper, lower = b, a

    upper_span = upper["col_spans"].get(int(col))
    lower_span = lower["col_spans"].get(int(col))
    upper_bottom = int(upper_span[1]) if upper_span is not None else int(upper["bbox"][2])
    lower_top = int(lower_span[0]) if lower_span is not None else int(lower["bbox"][0])
    start = upper_bottom + 1
    stop = lower_top - 1
    if start > stop:
        return None
    return int(start), int(stop)


def _horizontal_gap(a: dict, b: dict, row: int) -> tuple[int, int] | None:
    a_box = tuple(int(v) for v in a["bbox"])
    b_box = tuple(int(v) for v in b["bbox"])
    a_center = a_box[1] + a_box[3]
    b_center = b_box[1] + b_box[3]
    if (a_center, a_box[1], a_box[0]) <= (b_center, b_box[1], b_box[0]):
        left, right = a, b
    else:
        left, right = b, a

    left_span = left["row_spans"].get(int(row))
    right_span = right["row_spans"].get(int(row))
    left_edge = int(left_span[1]) if left_span is not None else int(left["bbox"][3])
    right_edge = int(right_span[0]) if right_span is not None else int(right["bbox"][1])
    start = left_edge + 1
    stop = right_edge - 1
    if start > stop:
        return None
    return int(start), int(stop)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect 4-connected foreground objects when their axis projections overlap.

    - Overlapping column projections create a vertical link at the midpoint column.
    - Overlapping row projections create a horizontal link at the midpoint row.
    - If both projections overlap, only the vertical link is drawn.
    - Original objects remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    objects = _find_objects(g)
    out = g.copy()

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a = objects[i]
            b = objects[j]
            ar0, ac0, ar1, ac1 = a["bbox"]
            br0, bc0, br1, bc1 = b["bbox"]
            col_overlap = _interval_overlap(ac0, ac1, bc0, bc1)
            row_overlap = _interval_overlap(ar0, ar1, br0, br1)
            color = _smaller_color(a, b)

            # When both bbox projections overlap, keep the spec's vertical priority
            # and span the open background corridor on the chosen midpoint column.
            if col_overlap is not None:
                col = _midpoint(*col_overlap)
                gap = _vertical_gap(a, b, col)
                if gap is None:
                    continue
                start, stop = gap
                for r in range(start, stop + 1):
                    if int(out[r, col]) in (bg, color):
                        out[r, col] = color
            elif row_overlap is not None:
                row = _midpoint(*row_overlap)
                gap = _horizontal_gap(a, b, row)
                if gap is None:
                    continue
                start, stop = gap
                for c in range(start, stop + 1):
                    if int(out[row, c]) in (bg, color):
                        out[row, c] = color

    return out


def _corner_l(height: int, width: int, *, top: bool, left: bool) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0 if top else height - 1, :] = True
    mask[:, 0 if left else width - 1] = True
    return mask


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    area = max(1, min(int(area), int(height * width)))
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    cells = {start}

    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        cell = random.choice(frontier)
        cells.add(cell)
        mask[cell] = True

    return mask


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(mask)
    rows = coords[:, 0] + int(top)
    cols = coords[:, 1] + int(left)
    assert np.all(grid[rows, cols] == BG)
    grid[rows, cols] = int(color)


def _place_corner_distractor(grid: np.ndarray, main_box: tuple[int, int, int, int], color: int) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = map(int, main_box)
    zones = [
        (1, r0 - 2, 1, c0 - 2),
        (1, r0 - 2, c1 + 2, w - 2),
        (r1 + 2, h - 2, 1, c0 - 2),
        (r1 + 2, h - 2, c1 + 2, w - 2),
    ]
    random.shuffle(zones)

    for zr0, zr1, zc0, zc1 in zones:
        if zr0 > zr1 or zc0 > zc1:
            continue
        zone_h = zr1 - zr0 + 1
        zone_w = zc1 - zc0 + 1
        for _ in range(30):
            shape_h = random.randint(1, min(3, zone_h))
            shape_w = random.randint(1, min(3, zone_w))
            area = random.randint(1, min(shape_h * shape_w, 5))
            mask = _random_connected_mask(shape_h, shape_w, area)
            top = random.randint(zr0, zr1 - shape_h + 1)
            left = random.randint(zc0, zc1 - shape_w + 1)
            coords = np.argwhere(mask)
            rows = coords[:, 0] + int(top)
            cols = coords[:, 1] + int(left)
            if np.all(grid[rows, cols] == BG):
                grid[rows, cols] = int(color)
                return True
    return False


def _pair_types(grid: np.ndarray) -> list[str]:
    objects = _find_objects(grid)
    types: list[str] = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            rel = _relation_type(objects[i], objects[j])
            if rel is not None:
                types.append(rel)
    return sorted(types)


def _make_vertical_scene(*, overlap_rows: bool, transpose: bool) -> np.ndarray | None:
    height = random.randint(13, 19)
    width = random.randint(14, 20)
    grid = np.full((height, width), BG, dtype=int)

    colors = random.sample(PALETTE, 4)
    top_color, bottom_color, extra_color = colors[:3]

    top_h = random.randint(3, 6)
    bottom_h = random.randint(3, 6)
    if top_h == bottom_h:
        bottom_h = 3 + ((bottom_h - 2) % 4)
        if top_h == bottom_h:
            bottom_h = 6 if top_h != 6 else 3

    if overlap_rows:
        box_w = random.choice((5, 7))
        col_shift_options = [shift for shift in (2, 4) if shift < box_w and box_w + shift <= width - 2]
        if not col_shift_options:
            return None
        col_shift = random.choice(col_shift_options)
        shift_max = min(top_h - 1, bottom_h - 1)
        if shift_max < 1:
            return None
        shift = random.randint(1, shift_max)
        total_h = max(top_h, shift + bottom_h)
        if total_h > height - 2:
            return None
        top_r = random.randint(1, height - total_h - 1)
        bottom_r = top_r + shift
        left = random.randint(1, width - (box_w + col_shift) - 1)
        flip = bool(random.getrandbits(1))
        if flip:
            top_left = left + col_shift
            bottom_left = left
            top_mask = _corner_l(top_h, box_w, top=True, left=False)
            bottom_mask = _corner_l(bottom_h, box_w, top=False, left=True)
        else:
            top_left = left
            bottom_left = left + col_shift
            top_mask = _corner_l(top_h, box_w, top=True, left=True)
            bottom_mask = _corner_l(bottom_h, box_w, top=False, left=False)
    else:
        box_w = random.choice((3, 5, 7))
        if box_w >= width - 2:
            return None
        gap = random.randint(2, 4)
        total_h = top_h + gap + bottom_h
        if total_h > height - 2:
            return None
        top_r = random.randint(1, height - total_h - 1)
        bottom_r = top_r + top_h + gap
        left = random.randint(1, width - box_w - 1)
        upper_left = bool(random.getrandbits(1))
        top_left = left
        bottom_left = left
        top_mask = _corner_l(top_h, box_w, top=True, left=upper_left)
        bottom_mask = _corner_l(bottom_h, box_w, top=False, left=not upper_left)

    _paint_mask(grid, top_mask, top_r, top_left, top_color)
    _paint_mask(grid, bottom_mask, bottom_r, bottom_left, bottom_color)

    main_box = (
        top_r,
        min(top_left, bottom_left),
        max(top_r + top_h - 1, bottom_r + bottom_h - 1),
        max(top_left, bottom_left) + box_w - 1,
    )
    if random.random() < 0.65:
        _place_corner_distractor(grid, main_box, extra_color)

    if transpose:
        grid = grid.T.copy()

    return grid


def _scene_matches_branch(grid: np.ndarray, expected_type: str) -> bool:
    pair_types = _pair_types(grid)
    if pair_types != [expected_type]:
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return not np.array_equal(out, grid)


def generate_input() -> np.ndarray:
    global _GENERATE_CALL_COUNT
    branch = _BRANCH_CYCLE[_GENERATE_CALL_COUNT % len(_BRANCH_CYCLE)]
    _GENERATE_CALL_COUNT += 1

    for _ in range(MAX_ATTEMPTS):
        if branch == "vertical":
            candidate = _make_vertical_scene(overlap_rows=False, transpose=False)
            expected = "vertical"
        elif branch == "horizontal":
            candidate = _make_vertical_scene(overlap_rows=False, transpose=True)
            expected = "horizontal"
        else:
            candidate = _make_vertical_scene(overlap_rows=True, transpose=False)
            expected = "both"

        if candidate is None:
            continue
        if _scene_matches_branch(candidate, expected):
            return candidate

    raise RuntimeError(f"generate_input: failed to build a {branch} scene after {MAX_ATTEMPTS} attempts")
