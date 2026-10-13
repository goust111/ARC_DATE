from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
DIRECTIONS = ("tl", "tr", "bl", "br")
BACKGROUND = 0
AREA_CHOICES = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
COLOR_COUNT_CHOICES = [2, 3, 4]
MAX_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _closest_rectangle(area: int) -> tuple[int, int]:
    best = (1, int(area))
    limit = int(math.isqrt(area))
    for a in range(1, limit + 1):
        if area % a == 0:
            best = (int(a), int(area // a))
    return best


def _direction_from_anchor(anchor: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    ar, ac = map(int, anchor)
    r0, c0, r1, c1 = map(int, bbox)
    if ar < r0 and ac < c0:
        return "tl"
    if ar < r0 and ac > c1:
        return "tr"
    if ar > r1 and ac < c0:
        return "bl"
    if ar > r1 and ac > c1:
        return "br"
    return None


def _rectangle_origin(anchor: tuple[int, int], height: int, width: int, direction: str) -> tuple[int, int]:
    ar, ac = map(int, anchor)
    if direction == "tl":
        return ar, ac
    if direction == "tr":
        return ar, ac - width + 1
    if direction == "bl":
        return ar - height + 1, ac
    if direction == "br":
        return ar - height + 1, ac - width + 1
    raise AssertionError(f"unknown direction {direction}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's scattered fragments into a filled rectangle whose corner is fixed by a singleton anchor.

    Input structure:
    - The grid has a dominant background color.
    - Each foreground color appears as several separated 4-connected fragments plus one isolated single-pixel anchor.
    - The anchor sits outside the fragment group's bounding box in one diagonal corner direction: top-left, top-right, bottom-left, or bottom-right.

    Rule:
    - For each foreground color, remove the anchor candidate and count the remaining same-color fragment pixels.
    - Factor that count into the closest-to-square rectangle dimensions (a x b with a <= b and minimal b-a).
    - Determine which corner direction the anchor occupies relative to the fragment-group bounding box.
    - Draw the filled a-by-b rectangle of that color so the anchor becomes the corresponding rectangle corner, and discard the original fragments.

    Key insight: Two independent clues must be combined per color: fragment area determines rectangle size, while the anchor's diagonal position determines which rectangle corner is pinned.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = np.full(g.shape, bg, dtype=int)

    for color in sorted(int(x) for x in np.unique(g) if int(x) != bg):
        comps = _components(g == color)
        valid: list[tuple[tuple[int, int], list[tuple[int, int]], str]] = []
        for comp in comps:
            if len(comp) != 1:
                continue
            anchor = comp[0]
            fragments = [cell for other in comps if other is not comp for cell in other]
            if not fragments:
                continue
            direction = _direction_from_anchor(anchor, _bbox(fragments))
            if direction is not None:
                valid.append((anchor, fragments, direction))

        assert len(valid) == 1, "each color must have exactly one valid singleton anchor"
        anchor, fragments, direction = valid[0]
        height, width = _closest_rectangle(len(fragments))
        r0, c0 = _rectangle_origin(anchor, height, width, direction)
        r1 = r0 + height
        c1 = c0 + width
        assert 0 <= r0 and 0 <= c0 and r1 <= g.shape[0] and c1 <= g.shape[1], "reassembled rectangle must fit"
        assert np.all(out[r0:r1, c0:c1] == bg), "reassembled rectangles must not overlap"
        out[r0:r1, c0:c1] = int(color)

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _, _ = _bbox(cells)
    return {(int(r - r0), int(c - c0)) for r, c in cells}


def _random_polyomino(size: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(cells) < size:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        r, c = cell
        for dr, dc in DIRS4:
            nxt = (int(r + dr), int(c + dc))
            if nxt not in cells:
                frontier.add(nxt)
    return _normalize(cells)


def _oriented_shape(size: int) -> set[tuple[int, int]]:
    cells = _random_polyomino(size)
    pts = list(cells)
    if random.random() < 0.5:
        pts = [(int(r), int(-c)) for r, c in pts]
    turns = random.randrange(4)
    for _ in range(turns):
        pts = [(int(c), int(-r)) for r, c in pts]
    return _normalize(set(pts))


def _partition_area(area: int, pieces: int) -> list[int]:
    parts = [2] * pieces
    for _ in range(area - 2 * pieces):
        parts[random.randrange(pieces)] += 1
    random.shuffle(parts)
    return parts


def _separated_from(existing: set[tuple[int, int]], cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for dr, dc in DIRS8:
            if (int(r + dr), int(c + dc)) in existing:
                return False
    return True


def _anchor_ranges(
    direction: str,
    frag_bbox: tuple[int, int, int, int],
    canvas_h: int,
    canvas_w: int,
    rect_h: int,
    rect_w: int,
) -> tuple[range, range]:
    r0, c0, r1, c1 = map(int, frag_bbox)
    if direction == "tl":
        row_hi = min(r0 - 2, canvas_h - rect_h)
        col_hi = min(c0 - 2, canvas_w - rect_w)
        return range(0, row_hi + 1), range(0, col_hi + 1)
    if direction == "tr":
        row_hi = min(r0 - 2, canvas_h - rect_h)
        col_lo = max(c1 + 2, rect_w - 1)
        return range(0, row_hi + 1), range(col_lo, canvas_w)
    if direction == "bl":
        row_lo = max(r1 + 2, rect_h - 1)
        col_hi = min(c0 - 2, canvas_w - rect_w)
        return range(row_lo, canvas_h), range(0, col_hi + 1)
    row_lo = max(r1 + 2, rect_h - 1)
    col_lo = max(c1 + 2, rect_w - 1)
    return range(row_lo, canvas_h), range(col_lo, canvas_w)


def _fragment_translation_bounds(direction: str, canvas_h: int, canvas_w: int, shape_h: int, shape_w: int) -> tuple[int, int, int, int]:
    if direction == "tl":
        return 2, canvas_h - shape_h, 2, canvas_w - shape_w
    if direction == "tr":
        return 2, canvas_h - shape_h, 0, canvas_w - shape_w - 2
    if direction == "bl":
        return 0, canvas_h - shape_h - 2, 2, canvas_w - shape_w
    return 0, canvas_h - shape_h - 2, 0, canvas_w - shape_w - 2


def _make_local_object(color: int) -> dict | None:
    area = random.choice(AREA_CHOICES)
    rect_h, rect_w = _closest_rectangle(area)
    piece_count = random.randint(2, min(4, area // 2))
    piece_sizes = sorted(_partition_area(area, piece_count), reverse=True)

    base = int(math.ceil(math.sqrt(area)))
    canvas_h = max(rect_h + 6, base + 6, piece_count * 3 + 3)
    canvas_w = max(rect_w + 6, base + 6, piece_count * 3 + 3)
    if canvas_h > 16 or canvas_w > 16:
        return None

    direction = random.choice(DIRECTIONS)
    placed: set[tuple[int, int]] = set()

    for size in piece_sizes:
        success = False
        for _ in range(100):
            shape = _oriented_shape(size)
            _, _, sr1, sc1 = _bbox(shape)
            shape_h = sr1 + 1
            shape_w = sc1 + 1
            r_lo, r_hi, c_lo, c_hi = _fragment_translation_bounds(direction, canvas_h, canvas_w, shape_h, shape_w)
            if r_lo > r_hi or c_lo > c_hi:
                return None
            off_r = random.randint(r_lo, r_hi)
            off_c = random.randint(c_lo, c_hi)
            cells = {(int(r + off_r), int(c + off_c)) for r, c in shape}
            if _separated_from(placed, cells):
                placed |= cells
                success = True
                break
        if not success:
            return None

    frag_bbox = _bbox(placed)
    row_range, col_range = _anchor_ranges(direction, frag_bbox, canvas_h, canvas_w, rect_h, rect_w)
    anchors = [(int(r), int(c)) for r in row_range for c in col_range]
    if not anchors:
        return None
    anchor = random.choice(anchors)

    input_cells = set(placed)
    input_cells.add(anchor)
    out_r0, out_c0 = _rectangle_origin(anchor, rect_h, rect_w, direction)
    output_cells = {
        (int(r), int(c))
        for r in range(out_r0, out_r0 + rect_h)
        for c in range(out_c0, out_c0 + rect_w)
    }

    all_cells = input_cells | output_cells
    min_r, min_c, max_r, max_c = _bbox(all_cells)
    shifted_input = {(int(r - min_r), int(c - min_c)) for r, c in input_cells}
    shifted_output = {(int(r - min_r), int(c - min_c)) for r, c in output_cells}
    return {
        "color": int(color),
        "direction": direction,
        "shape_h": int(max_r - min_r + 1),
        "shape_w": int(max_c - min_c + 1),
        "input_cells": shifted_input,
        "output_cells": shifted_output,
    }


def _grid_bounds(objects: list[dict]) -> tuple[int, int]:
    n = len(objects)
    lower = {2: 14, 3: 18, 4: 22}[n]
    max_h = max(obj["shape_h"] for obj in objects)
    max_w = max(obj["shape_w"] for obj in objects)
    h_lo = max(lower, max_h + 1)
    w_lo = max(lower, max_w + 1)
    if h_lo > 30 or w_lo > 30:
        raise ValueError("object too large for ARC grid")
    return random.randint(h_lo, 30), random.randint(w_lo, 30)


def _try_place_objects(objects: list[dict], height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    expected = np.full((height, width), BACKGROUND, dtype=int)
    input_occ: set[tuple[int, int]] = set()
    output_occ: set[tuple[int, int]] = set()

    order = sorted(objects, key=lambda obj: obj["shape_h"] * obj["shape_w"], reverse=True)
    for obj in order:
        shape_h = int(obj["shape_h"])
        shape_w = int(obj["shape_w"])
        if shape_h > height or shape_w > width:
            return None
        placed = False
        for _ in range(250):
            off_r = random.randint(0, height - shape_h)
            off_c = random.randint(0, width - shape_w)
            translated_input = {(int(r + off_r), int(c + off_c)) for r, c in obj["input_cells"]}
            translated_output = {(int(r + off_r), int(c + off_c)) for r, c in obj["output_cells"]}
            if not _separated_from(input_occ, translated_input):
                continue
            if not _separated_from(output_occ, translated_output):
                continue
            for r, c in translated_input:
                grid[r, c] = int(obj["color"])
            for r, c in translated_output:
                expected[r, c] = int(obj["color"])
            input_occ |= translated_input
            output_occ |= translated_output
            placed = True
            break
        if not placed:
            return None
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    color_count = random.choice(COLOR_COUNT_CHOICES)
    colors = random.sample(list(range(1, 10)), color_count)
    objects: list[dict] = []
    for color in colors:
        obj = _make_local_object(color)
        if obj is None:
            return None
        objects.append(obj)

    height, width = _grid_bounds(objects)
    placed = _try_place_objects(objects, height, width)
    if placed is None:
        return None
    grid, expected = placed

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
