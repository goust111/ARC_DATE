import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
GREY = 5
MAROON = 9
BODY_COLORS = (1, 4, 6, 7, 8)
DIRS = ("up", "down", "left", "right")
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _scale_index(idx: int, src_len: int, dst_len: int) -> int:
    if src_len <= 1 or dst_len <= 1:
        return 0
    return int(round(idx * (dst_len - 1) / (src_len - 1)))


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == MAROON)]
    assert len(cols) == 1, "expected exactly one maroon separator column"
    return int(cols[0])


def _find_canvas_bbox(grid: np.ndarray, sep_c: int) -> tuple[int, int, int, int]:
    left = grid[:, :sep_c]
    coords = np.argwhere(left == GREY)
    assert coords.size > 0, "expected a grey canvas on the left"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    canvas = left[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(canvas == GREY), "canvas must be a solid grey rectangle"
    return int(r0), int(c0), int(r1), int(c1)


def _direction_from_markers(anchor: tuple[int, int], marker: tuple[int, int]) -> str:
    ar, ac = anchor
    gr, gc = marker
    if gc == ac and gr < ar:
        return "up"
    if gc == ac and gr > ar:
        return "down"
    if gr == ar and gc < ac:
        return "left"
    if gr == ar and gc > ac:
        return "right"
    raise AssertionError("green marker must be strictly above/below/left/right of the red anchor")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each right-side object's body onto the left grey canvas.

    The red pixel is the anchor used for boundary alignment and ratio mapping.
    The green pixel gives which canvas edge receives the stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_c = _find_separator_col(g)
    canvas_r0, canvas_c0, canvas_r1, canvas_c1 = _find_canvas_bbox(g, sep_c)
    canvas_h = canvas_r1 - canvas_r0 + 1
    canvas_w = canvas_c1 - canvas_c0 + 1
    out = np.full((canvas_h, canvas_w), GREY, dtype=int)

    right_mask = (np.arange(g.shape[1])[None, :] > sep_c) & (g != BG)
    right_mask &= g != MAROON
    for coords in _components(right_mask):
        values = [int(g[r, c]) for r, c in coords]
        anchors = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == RED]
        markers = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == GREEN]
        body = [(int(r), int(c)) for r, c in coords if int(g[r, c]) not in (RED, GREEN)]

        assert len(anchors) == 1, "each object must contain exactly one red anchor"
        assert len(markers) == 1, "each object must contain exactly one green direction marker"
        assert body, "each object must have a non-empty body"
        body_colors = {int(g[r, c]) for r, c in body}
        assert len(body_colors) == 1, "each object body must be monochromatic"

        anchor = anchors[0]
        marker = markers[0]
        direction = _direction_from_markers(anchor, marker)

        r0, c0, r1, c1 = _bbox(coords)
        obj_h = r1 - r0 + 1
        obj_w = c1 - c0 + 1
        ar_rel = anchor[0] - r0
        ac_rel = anchor[1] - c0

        if direction in ("up", "down"):
            target_c = canvas_c0 + _scale_index(ac_rel, obj_w, canvas_w)
            target_r = canvas_r0 if direction == "up" else canvas_r1
        else:
            target_r = canvas_r0 + _scale_index(ar_rel, obj_h, canvas_h)
            target_c = canvas_c0 if direction == "left" else canvas_c1

        dr = target_r - anchor[0]
        dc = target_c - anchor[1]
        for br, bc in body:
            rr = br + dr
            cc = bc + dc
            if canvas_r0 <= rr <= canvas_r1 and canvas_c0 <= cc <= canvas_c1:
                out[rr - canvas_r0, cc - canvas_c0] = BG

    return out


def _grow_connected(allowed: set[tuple[int, int]], seeds: set[tuple[int, int]], target: int) -> set[tuple[int, int]] | None:
    cells = set(seeds)
    if not cells <= allowed:
        return None
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in allowed and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _render_object(
    body_cells: set[tuple[int, int]],
    anchor: tuple[int, int],
    marker: tuple[int, int],
    body_color: int,
) -> np.ndarray:
    all_cells = set(body_cells) | {anchor, marker}
    coords = np.array(sorted(all_cells), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    obj = np.full((h, w), BG, dtype=int)
    for r, c in body_cells:
        obj[r - r0, c - c0] = body_color
    obj[anchor[0] - r0, anchor[1] - c0] = RED
    obj[marker[0] - r0, marker[1] - c0] = GREEN
    return obj


def _generate_oriented_object(direction: str, max_h: int, max_w: int, body_color: int) -> np.ndarray | None:
    max_h = min(max_h, 7)
    max_w = min(max_w, 7)
    if max_h < 4 or max_w < 4:
        return None

    for _ in range(120):
        h = random.randint(4, max_h)
        w = random.randint(4, max_w)

        if direction == "up":
            ac = random.randint(1, w - 2)
            anchor = (1, ac)
            marker = (0, ac)
            allowed = {(r, c) for r in range(1, h) for c in range(w)} - {anchor}
            seeds = {(1, ac - 1), (1, ac + 1), (2, ac)}
            if ac > 1 and random.random() < 0.7:
                seeds.add((2, ac - 1))
            if ac < w - 2 and random.random() < 0.7:
                seeds.add((2, ac + 1))
        elif direction == "down":
            ac = random.randint(1, w - 2)
            anchor = (h - 2, ac)
            marker = (h - 1, ac)
            allowed = {(r, c) for r in range(h - 1) for c in range(w)} - {anchor}
            seeds = {(h - 2, ac - 1), (h - 2, ac + 1), (h - 3, ac)}
            if ac > 1 and random.random() < 0.7:
                seeds.add((h - 3, ac - 1))
            if ac < w - 2 and random.random() < 0.7:
                seeds.add((h - 3, ac + 1))
        elif direction == "left":
            ar = random.randint(1, h - 2)
            anchor = (ar, 1)
            marker = (ar, 0)
            allowed = {(r, c) for r in range(h) for c in range(1, w)} - {anchor}
            seeds = {(ar - 1, 1), (ar + 1, 1), (ar, 2)}
            if ar > 1 and random.random() < 0.7:
                seeds.add((ar - 1, 2))
            if ar < h - 2 and random.random() < 0.7:
                seeds.add((ar + 1, 2))
        else:
            ar = random.randint(1, h - 2)
            anchor = (ar, w - 2)
            marker = (ar, w - 1)
            allowed = {(r, c) for r in range(h) for c in range(w - 1)} - {anchor}
            seeds = {(ar - 1, w - 2), (ar + 1, w - 2), (ar, w - 3)}
            if ar > 1 and random.random() < 0.7:
                seeds.add((ar - 1, w - 3))
            if ar < h - 2 and random.random() < 0.7:
                seeds.add((ar + 1, w - 3))

        allowed_area = len(allowed)
        min_area = max(6, len(seeds))
        max_area = min(allowed_area, 14)
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)
        body_cells = _grow_connected(allowed, seeds, target_area)
        if body_cells is None:
            continue

        body_coords = np.array(sorted(body_cells), dtype=int)
        br0, bc0, br1, bc1 = _bbox(body_coords)
        if len(body_cells) == (br1 - br0 + 1) * (bc1 - bc0 + 1):
            continue

        return _render_object(body_cells, anchor, marker, body_color)

    return None


def _direction_list(n: int) -> list[str]:
    base = list(DIRS)
    random.shuffle(base)
    if n <= 4:
        return base[:n]
    out = base[:]
    while len(out) < n:
        out.append(random.choice(DIRS))
    random.shuffle(out)
    return out


def _paste_nonzero(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = src.shape
    for r in range(h):
        for c in range(w):
            if src[r, c] != BG:
                dst[r0 + r, c0 + c] = int(src[r, c])


def _place_objects(
    grid: np.ndarray,
    sep_c: int,
    right_margin: int,
    objects: list[np.ndarray],
) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(objects)), key=lambda i: int(np.count_nonzero(objects[i])), reverse=True)

    for idx in order:
        obj = objects[idx]
        oh, ow = obj.shape
        placed = False
        row_lo, row_hi = 1, h - oh - 1
        col_lo, col_hi = sep_c + 2, w - right_margin - ow
        if row_lo > row_hi or col_lo > col_hi:
            return False

        candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + oh + 1)
            cc1 = min(w, c0 + ow + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            _paste_nonzero(grid, obj, r0, c0)
            occupied[r0 : r0 + oh, c0 : c0 + ow] = True
            placed = True
            break

        if not placed:
            return False

    return True


def _build_input() -> np.ndarray | None:
    grid_h = random.randint(17, 22)
    canvas_h = random.randint(8, min(12, grid_h - 4))
    canvas_w = random.randint(8, 10)
    top_margin = random.randint(1, grid_h - canvas_h - 1)
    left_margin = random.randint(1, 2)
    gap_before_sep = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    right_w = random.randint(10, 13)
    sep_c = left_margin + canvas_w + gap_before_sep
    grid_w = sep_c + 1 + right_w + right_margin
    if grid_w > 30:
        return None

    n_objects = random.randint(2, 5)
    directions = _direction_list(n_objects)
    colors = random.sample(BODY_COLORS, n_objects)

    objects = []
    for direction, color in zip(directions, colors):
        obj = _generate_oriented_object(direction, canvas_h, canvas_w, color)
        if obj is None:
            return None
        objects.append(obj)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[:, sep_c] = MAROON
    grid[top_margin : top_margin + canvas_h, left_margin : left_margin + canvas_w] = GREY

    if not _place_objects(grid, sep_c, right_margin, objects):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != (canvas_h, canvas_w):
        return None
    if not (np.any(out == BG) and np.any(out == GREY)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
